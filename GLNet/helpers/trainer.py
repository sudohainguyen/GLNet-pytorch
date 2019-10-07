import numpy as np
import torch
import torch.nn.functional as F

from ..models.functional import crop_global
from ..utils import PhaseMode
from ..utils.metrics import ConfusionMatrix
from ..utils.processing import (
    images_transform,
    masks_transform,
    resize,
    global2patch,
    patch2global
)

class Trainer(object):

    def __init__(
        self,
        criterion,
        optimizer,
        n_class,
        size_g,
        size_p,
        sub_batch_size=6,
        mode=PhaseMode.GlobalOnly,
        lamb_fmreg=0.15,
    ):
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics_global = ConfusionMatrix(n_class)
        self.metrics_local = ConfusionMatrix(n_class)
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class

        self.size_g = size_g
        self.size_p = size_p

        self.sub_batch_size = sub_batch_size
        self.mode = mode
        self.lamb_fmreg = lamb_fmreg

    def set_train(self, model):
        model.module.ensemble_conv.train()
        if self.mode is PhaseMode.GlobalOnly or self.mode is PhaseMode.GlobalFromLocal:
            model.module.backbone_global.train()
            model.module.fpn_global.train()
        else:
            model.module.backbone_local.train()
            model.module.fpn_local.train()

    def get_scores(self):
        score_train = self.metrics.get_scores()
        score_train_local = self.metrics_local.get_scores()
        score_train_global = self.metrics_global.get_scores()
        return score_train, score_train_global, score_train_local

    def reset_metrics(self):
        self.metrics.reset()
        self.metrics_local.reset()
        self.metrics_global.reset()

    def train(self, sample, model, global_fixed):
        images, labels = sample["image"], sample["label"]  # PIL images
        # lbls = [RGB_mapping_to_class(np.array(label)) for label in labels]
        # labels = [Image.fromarray(lbl) for lbl in lbls]
        # del lbls

        labels_npy = masks_transform(
            labels, numpy=True
        )  # label of origin size in numpy

        images_glb = resize(images, self.size_g)  # list of resized PIL images
        images_glb = images_transform(images_glb)

        labels_glb = resize(labels, (self.size_g[0] // 4, self.size_g[1] // 4), is_mask=True) # FPN down 1/4, for loss
        labels_glb = masks_transform(labels_glb)  # 127 * 127 * 8 = 129032

        if self.mode is PhaseMode.LocalFromGlobal or self.mode is PhaseMode.GlobalFromLocal:
            patches, coordinates, templates, sizes, ratios = global2patch(
            # patches, coordinates, _, sizes, ratios = global2patch(
                images, self.size_p
            )
            label_patches, _, _, _, _ = global2patch(labels, self.size_p)
            
            predicted_patches = [
                np.zeros(
                    (len(coordinates[i]), self.n_class, self.size_p[0], self.size_p[1])
                )
                for i in range(len(images))
            ]
            predicted_ensembles = [
                np.zeros(
                    (len(coordinates[i]), self.n_class, self.size_p[0], self.size_p[1])
                )
                for i in range(len(images))
            ]
            outputs_global = [None for i in range(len(images))]

        if self.mode is PhaseMode.GlobalOnly:
            # training with only (resized) global image 
            outputs_global, _ = model.forward(images_glb, None, None, None)
            loss = self.criterion(outputs_global, labels_glb)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            

        if self.mode is PhaseMode.LocalFromGlobal:
            # training with patches 
            for i in range(len(images)):
                j = 0
                # while j < self.n**2:
                while j < len(coordinates[i]):
                    patches_var = images_transform(
                        patches[i][j : j + self.sub_batch_size]
                    )  # b, c, h, w
                    label_patches_var = masks_transform(
                        resize(
                            label_patches[i][j : j + self.sub_batch_size],
                            (self.size_p[0] // 4, self.size_p[1] // 4),
                            is_mask=True,
                        )
                    )  # down 1/4 for loss
                    # label_patches_var = masks_transform(label_patches[i][j : j+self.sub_batch_size])

                    # output_ensembles, output_global, output_patches, fmreg_l2 = model.forward(images_glb[i:i+1], patches_var, self.coordinates[j : j+self.sub_batch_size], self.ratio, mode=self.mode, n_patch=self.n**2) # include cordinates
                    output_ensembles, output_global, output_patches, fmreg_l2 = model.forward(
                        images_glb[i : i + 1],
                        patches_var,
                        coordinates[i][j : j + self.sub_batch_size],
                        ratios[i],
                        mode=self.mode,
                        n_patch=len(coordinates[i]),
                    )
                    loss = (
                        self.criterion(output_patches, label_patches_var)
                        + self.criterion(output_ensembles, label_patches_var)
                        + self.lamb_fmreg * fmreg_l2
                    )
                    loss.backward()

                    # patch predictions
                    predicted_patches[i][j : j + output_patches.size()[0]] = (
                        F.interpolate(output_patches, size=self.size_p, mode="nearest")
                        .data.cpu()
                        .numpy()
                    )
                    predicted_ensembles[i][j : j + output_ensembles.size()[0]] = (
                        F.interpolate(
                            output_ensembles, size=self.size_p, mode="nearest"
                        )
                        .data.cpu()
                        .numpy()
                    )
                    j += self.sub_batch_size
                outputs_global[i] = output_global
            outputs_global = torch.cat(outputs_global, dim=0)

            self.optimizer.step()
            self.optimizer.zero_grad()
            

        if self.mode is PhaseMode.GlobalFromLocal:
            # train global with help from patches 
            # go through local patches to collect feature maps
            # collect predictions from patches
            for i in range(len(images)):
                j = 0
                while j < len(coordinates[i]):
                    patches_var = images_transform(
                        patches[i][j : j + self.sub_batch_size]
                    )  # b, c, h, w
                    # fm_patches, output_patches = model.module.collect_local_fm(images_glb[i:i+1], patches_var, self.ratio, self.coordinates, [j, j+self.sub_batch_size], len(images), global_model=global_fixed, template=self.template, n_patch_all=self.n**2) # include cordinates
                    fm_patches, output_patches = model.module.collect_local_fm(
                        images_glb[i : i + 1],
                        patches_var,
                        ratios[i],
                        coordinates[i],
                        [j, j + self.sub_batch_size],
                        len(images),
                        global_model=global_fixed,
                        template=templates[i],
                        n_patch_all=len(coordinates[i]),
                    )
                    predicted_patches[i][j : j + output_patches.size()[0]] = (
                        F.interpolate(output_patches, size=self.size_p, mode="nearest")
                        .data.cpu()
                        .numpy()
                    )
                    j += self.sub_batch_size
            # train on global image
            outputs_global, fm_global = model.forward(
                images_glb, None, None, None, mode=self.mode
            )
            loss = self.criterion(outputs_global, labels_glb)
            loss.backward(retain_graph=True)
            # fmreg loss
            # generate ensembles & calc loss
            for i in range(len(images)):
                j = 0
                # while j < self.n**2:
                while j < len(coordinates[i]):
                    label_patches_var = masks_transform(
                        resize(
                            label_patches[i][j : j + self.sub_batch_size],
                            (self.size_p[0] // 4, self.size_p[1] // 4),
                            is_mask=True,
                        )
                    )
                    # label_patches_var = masks_transform(resize(label_patches[i][j : j+self.sub_batch_size], self.size_p, label=True))
                    fl = fm_patches[i][j : j + self.sub_batch_size].cuda()
                    # fg = model.module._crop_global(fm_global[i:i+1], self.coordinates[j:j+self.sub_batch_size], self.ratio)[0]
                    fg = crop_global(
                        fm_global[i : i + 1],
                        coordinates[i][j : j + self.sub_batch_size],
                        ratios[i],
                    )[0]
                    fg = F.interpolate(fg, size=fl.size()[2:], mode="bilinear")
                    output_ensembles = model.module.ensemble(fl, fg)
                    # output_ensembles = F.interpolate(model.module.ensemble(fl, fg), self.size_p, **model.module._up_kwargs)
                    loss = self.criterion(
                        output_ensembles, label_patches_var
                    )  # + 0.15 * mse(fl, fg)
                    # if i == len(images) - 1 and j + self.sub_batch_size >= self.n**2:
                    if i == len(images) - 1 and j + self.sub_batch_size >= len(
                        coordinates[i]
                    ):
                        loss.backward()
                    else:
                        loss.backward(retain_graph=True)

                    # ensemble predictions
                    predicted_ensembles[i][j : j + output_ensembles.size()[0]] = (
                        F.interpolate(
                            output_ensembles, size=self.size_p, mode="nearest"
                        )
                        .data.cpu()
                        .numpy()
                    )
                    j += self.sub_batch_size
            self.optimizer.step()
            self.optimizer.zero_grad()

        # global predictions 
        # predictions_global = F.interpolate(outputs_global.cpu(), self.size0, mode='nearest').argmax(1).detach().numpy()
        outputs_global = outputs_global.cpu()
        predictions_global = [
            F.interpolate(
                outputs_global[i : i + 1], images[i].size[::-1], mode="nearest"
            )
            .argmax(1)
            .detach()
            .numpy()
            for i in range(len(images))
        ]
        self.metrics_global.update(labels_npy, predictions_global)

        if self.mode is PhaseMode.LocalFromGlobal or self.mode is PhaseMode.GlobalFromLocal:
            # patch predictions 
            # merge softmax scores from patches (overlaps)
            scores_local = np.array(patch2global(predicted_patches, self.n_class, sizes, coordinates, self.size_p))  
            predictions_local = scores_local.argmax(1)  # b, h, w
            self.metrics_local.update(labels_npy, predictions_local)
            
            # combined/ensemble predictions 
            # scores = np.array(patch2global(predicted_ensembles, self.n_class, self.n, self.step, self.size0, self.size_p, len(images))) # merge softmax scores from patches (overlaps)
            scores = np.array(
                patch2global(
                    predicted_ensembles, self.n_class, sizes, coordinates, self.size_p
                )
            )  # merge softmax scores from patches (overlaps)
            predictions = scores.argmax(1)  # b, h, w
            self.metrics.update(labels_npy, predictions)
        return loss
