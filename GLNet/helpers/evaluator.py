import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
# from PIL import Image

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


class Evaluator(object):
    def __init__(self, n_class, size_g, size_p, sub_batch_size=6, mode=PhaseMode.GlobalOnly, test=False):
        self.metrics_global = ConfusionMatrix(n_class)
        self.metrics_local = ConfusionMatrix(n_class)
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class
    
        self.size_g = size_g
        self.size_p = size_p

        self.sub_batch_size = sub_batch_size
        self.mode = mode
        self.test = test

        if test:
            self.flip_range = [False, True]
            self.rotate_range = [0, 1, 2, 3]
        else:
            self.flip_range = [False]
            self.rotate_range = [0]

    def get_scores(self):
        score_train = self.metrics.get_scores()
        score_train_local = self.metrics_local.get_scores()
        score_train_global = self.metrics_global.get_scores()
        return score_train, score_train_global, score_train_local

    def reset_metrics(self):
        self.metrics.reset()
        self.metrics_local.reset()
        self.metrics_global.reset()

    def eval_test(self, sample, model, global_fixed):
        with torch.no_grad():
            images = sample["image"]
            if not self.test:
                labels = sample["label"]  # PIL images
                # lbls = [RGB_mapping_to_class(np.array(label)) for label in labels]
                # labels = [Image.fromarray(lbl) for lbl in lbls]
                labels_npy = masks_transform(labels, numpy=True)

            images_global = resize(images, self.size_g)
            outputs_global = np.zeros(
                (len(images), self.n_class, self.size_g[0] // 4, self.size_g[1] // 4)
            )
            # outputs_global = np.zeros((len(images), self.n_class, self.size_g[0], self.size_g[1]))
            if self.mode is PhaseMode.LocalFromGlobal or self.mode is PhaseMode.GlobalFromLocal:
                images_local = [image.copy() for image in images]
                # scores_local = np.zeros((len(images), self.n_class, self.size0[0], self.size0[1]))
                # scores = np.zeros((len(images), self.n_class, self.size0[0], self.size0[1]))
                scores_local = [
                    np.zeros((1, self.n_class, images[i].size[1], images[i].size[0]))
                    for i in range(len(images))
                ]
                scores = [
                    np.zeros((1, self.n_class, images[i].size[1], images[i].size[0]))
                    for i in range(len(images))
                ]

            for flip in self.flip_range:
                if flip:
                    # we already rotated images for 270'
                    for b in range(len(images)):
                        images_global[b] = transforms.functional.rotate(
                            images_global[b], 90
                        )  # rotate back!
                        images_global[b] = transforms.functional.hflip(images_global[b])
                        if self.mode is PhaseMode.LocalFromGlobal or self.mode is PhaseMode.GlobalFromLocal:
                            images_local[b] = transforms.functional.rotate(
                                images_local[b], 90
                            )  # rotate back!
                            images_local[b] = transforms.functional.hflip(
                                images_local[b]
                            )
                for angle in self.rotate_range:
                    if angle > 0:
                        for b in range(len(images)):
                            images_global[b] = transforms.functional.rotate(
                                images_global[b], 90
                            )
                            if self.mode is PhaseMode.LocalFromGlobal or self.mode is PhaseMode.GlobalFromLocal:
                                images_local[b] = transforms.functional.rotate(
                                    images_local[b], 90
                                )

                    # prepare global images onto cuda
                    images_glb = images_transform(images_global)  # b, c, h, w

                    if self.mode is PhaseMode.LocalFromGlobal or self.mode is PhaseMode.GlobalFromLocal:
                        # patches = global2patch(images_local, self.n, self.step, self.size_p)
                        patches, coordinates, templates, sizes, ratios = global2patch(
                        # patches, coordinates, _, sizes, ratios = global2patch(
                            images, self.size_p
                        )
                        # predicted_patches = [ np.zeros((self.n**2, self.n_class, self.size_p[0], self.size_p[1])) for i in range(len(images)) ]
                        # predicted_ensembles = [ np.zeros((self.n**2, self.n_class, self.size_p[0], self.size_p[1])) for i in range(len(images)) ]
                        predicted_patches = [
                            np.zeros(
                                (
                                    len(coordinates[i]),
                                    self.n_class,
                                    self.size_p[0],
                                    self.size_p[1],
                                )
                            )
                            for i in range(len(images))
                        ]
                        predicted_ensembles = [
                            np.zeros(
                                (
                                    len(coordinates[i]),
                                    self.n_class,
                                    self.size_p[0],
                                    self.size_p[1],
                                )
                            )
                            for i in range(len(images))
                        ]

                    if self.mode is PhaseMode.GlobalOnly:
                        # eval with only resized global image 
                        if flip:
                            outputs_global += np.flip(
                                np.rot90(
                                    model.forward(images_glb, None, None, None)[0]
                                    .data.cpu()
                                    .numpy(),
                                    k=angle,
                                    axes=(3, 2),
                                ),
                                axis=3,
                            )
                        else:
                            outputs_global += np.rot90(
                                model.forward(images_glb, None, None, None)[0]
                                .data.cpu()
                                .numpy(),
                                k=angle,
                                axes=(3, 2),
                            )

                    if self.mode is PhaseMode.LocalFromGlobal:
                        # eval with patches 
                        for i in range(len(images)):
                            j = 0
                            # while j < self.n**2:
                            while j < len(coordinates[i]):
                                patches_var = images_transform(
                                    patches[i][j : j + self.sub_batch_size]
                                )  # b, c, h, w
                                # output_ensembles, output_global, output_patches, _ = model.forward(images_glb[i:i+1], patches_var, self.coordinates[j : j+self.sub_batch_size], self.ratio, mode=self.mode, n_patch=self.n**2) # include cordinates
                                output_ensembles, output_global, output_patches, _ = model.forward(
                                    images_glb[i : i + 1],
                                    patches_var,
                                    coordinates[i][j : j + self.sub_batch_size],
                                    ratios[i],
                                    mode=self.mode,
                                    n_patch=len(coordinates[i]),
                                )

                                # patch predictions
                                predicted_patches[i][
                                    j : j + output_patches.size()[0]
                                ] += (
                                    F.interpolate(
                                        output_patches, size=self.size_p, mode="nearest"
                                    )
                                    .data.cpu()
                                    .numpy()
                                )
                                predicted_ensembles[i][
                                    j : j + output_ensembles.size()[0]
                                ] += (
                                    F.interpolate(
                                        output_ensembles,
                                        size=self.size_p,
                                        mode="nearest",
                                    )
                                    .data.cpu()
                                    .numpy()
                                )
                                j += patches_var.size()[0]
                            if flip:
                                outputs_global[i] += np.flip(
                                    np.rot90(
                                        output_global[0].data.cpu().numpy(),
                                        k=angle,
                                        axes=(2, 1),
                                    ),
                                    axis=2,
                                )
                                # scores_local[i] += np.flip(np.rot90(np.array(patch2global(predicted_patches[i:i+1], self.n_class, self.n, self.step, self.size0, self.size_p, len(images))), k=angle, axes=(3, 2)), axis=3) # merge softmax scores from patches (overlaps)
                                # scores[i] += np.flip(np.rot90(np.array(patch2global(predicted_ensembles[i:i+1], self.n_class, self.n, self.step, self.size0, self.size_p, len(images))), k=angle, axes=(3, 2)), axis=3) # merge softmax scores from patches (overlaps)
                                scores_local[i] += np.flip(
                                    np.rot90(
                                        np.array(
                                            patch2global(
                                                predicted_patches[i : i + 1],
                                                self.n_class,
                                                sizes[i : i + 1],
                                                coordinates[i : i + 1],
                                                self.size_p,
                                            )
                                        ),
                                        k=angle,
                                        axes=(3, 2),
                                    ),
                                    axis=3,
                                )  # merge softmax scores from patches (overlaps)
                                scores[i] += np.flip(
                                    np.rot90(
                                        np.array(
                                            patch2global(
                                                predicted_ensembles[i : i + 1],
                                                self.n_class,
                                                sizes[i : i + 1],
                                                coordinates[i : i + 1],
                                                self.size_p,
                                            )
                                        ),
                                        k=angle,
                                        axes=(3, 2),
                                    ),
                                    axis=3,
                                )  # merge softmax scores from patches (overlaps)
                            else:
                                outputs_global[i] += np.rot90(
                                    output_global[0].data.cpu().numpy(),
                                    k=angle,
                                    axes=(2, 1),
                                )
                                # scores_local[i] += np.rot90(np.array(patch2global(predicted_patches[i:i+1], self.n_class, self.n, self.step, self.size0, self.size_p, len(images))), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                                # scores[i] += np.rot90(np.array(patch2global(predicted_ensembles[i:i+1], self.n_class, self.n, self.step, self.size0, self.size_p, len(images))), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                                scores_local[i] += np.rot90(
                                    np.array(
                                        patch2global(
                                            predicted_patches[i : i + 1],
                                            self.n_class,
                                            sizes[i : i + 1],
                                            coordinates[i : i + 1],
                                            self.size_p,
                                        )
                                    ),
                                    k=angle,
                                    axes=(3, 2),
                                )  # merge softmax scores from patches (overlaps)
                                scores[i] += np.rot90(
                                    np.array(
                                        patch2global(
                                            predicted_ensembles[i : i + 1],
                                            self.n_class,
                                            sizes[i : i + 1],
                                            coordinates[i : i + 1],
                                            self.size_p,
                                        )
                                    ),
                                    k=angle,
                                    axes=(3, 2),
                                )  # merge softmax scores from patches (overlaps)

                    if self.mode is PhaseMode.GlobalFromLocal:
                        # eval global with help from patches 
                        # go through local patches to collect feature maps
                        # collect predictions from patches
                        for i in range(len(images)):
                            j = 0
                            # while j < self.n**2:
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
                                predicted_patches[i][
                                    j : j + output_patches.size()[0]
                                ] += (
                                    F.interpolate(
                                        output_patches, size=self.size_p, mode="nearest"
                                    )
                                    .data.cpu()
                                    .numpy()
                                )
                                j += self.sub_batch_size
                        # go through global image
                        # tmp, fm_global = model.forward(images_glb, None, self.coordinates, self.ratio, mode=self.mode, global_model=None, n_patch=self.n**2) # include cordinates
                        tmp, fm_global = model.forward(
                            images_glb, None, None, None, mode=self.mode
                        )
                        if flip:
                            outputs_global += np.flip(
                                np.rot90(tmp.data.cpu().numpy(), k=angle, axes=(3, 2)),
                                axis=3,
                            )
                        else:
                            outputs_global += np.rot90(
                                tmp.data.cpu().numpy(), k=angle, axes=(3, 2)
                            )
                        # generate ensembles
                        for i in range(len(images)):
                            j = 0
                            # while j < self.n ** 2:
                            while j < len(coordinates[i]):
                                fl = fm_patches[i][j : j + self.sub_batch_size].cuda()
                                # fg = model.module._crop_global(fm_global[i:i+1], self.coordinates[j:j+self.sub_batch_size], self.ratio)[0]
                                fg = crop_global(
                                    fm_global[i : i + 1],
                                    coordinates[i][j : j + self.sub_batch_size],
                                    ratios[i])[0]
                                fg = F.interpolate(
                                    fg, size=fl.size()[2:], mode="bilinear"
                                )
                                output_ensembles = model.module.ensemble(
                                    fl, fg
                                )  # include cordinates
                                # output_ensembles = F.interpolate(model.module.ensemble(fl, fg), self.size_p, **model.module._up_kwargs)

                                # ensemble predictions
                                predicted_ensembles[i][
                                    j : j + output_ensembles.size()[0]
                                ] += (
                                    F.interpolate(
                                        output_ensembles,
                                        size=self.size_p,
                                        mode="nearest",
                                    )
                                    .data.cpu()
                                    .numpy()
                                )
                                j += self.sub_batch_size
                            if flip:
                                # scores_local[i] += np.flip(np.rot90(np.array(patch2global(predicted_patches[i:i+1], self.n_class, self.n, self.step, self.size0, self.size_p, len(images))), k=angle, axes=(3, 2)), axis=3) # merge softmax scores from patches (overlaps)
                                # scores[i] += np.flip(np.rot90(np.array(patch2global(predicted_ensembles[i:i+1], self.n_class, self.n, self.step, self.size0, self.size_p, len(images))), k=angle, axes=(3, 2)), axis=3) # merge softmax scores from patches (overlaps)
                                scores_local[i] += np.flip(
                                    np.rot90(
                                        np.array(
                                            patch2global(
                                                predicted_patches[i : i + 1],
                                                self.n_class,
                                                sizes[i : i + 1],
                                                coordinates[i : i + 1],
                                                self.size_p,
                                            )
                                        ),
                                        k=angle,
                                        axes=(3, 2),
                                    ),
                                    axis=3,
                                )[
                                    0
                                ]  # merge softmax scores from patches (overlaps)
                                scores[i] += np.flip(
                                    np.rot90(
                                        np.array(
                                            patch2global(
                                                predicted_ensembles[i : i + 1],
                                                self.n_class,
                                                sizes[i : i + 1],
                                                coordinates[i : i + 1],
                                                self.size_p,
                                            )
                                        ),
                                        k=angle,
                                        axes=(3, 2),
                                    ),
                                    axis=3,
                                )[
                                    0
                                ]  # merge softmax scores from patches (overlaps)
                            else:
                                # scores_local[i] += np.rot90(np.array(patch2global(predicted_patches[i:i+1], self.n_class, self.n, self.step, self.size0, self.size_p, len(images))), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                                # scores[i] += np.rot90(np.array(patch2global(predicted_ensembles[i:i+1], self.n_class, self.n, self.step, self.size0, self.size_p, len(images))), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                                scores_local[i] += np.rot90(
                                    np.array(
                                        patch2global(
                                            predicted_patches[i : i + 1],
                                            self.n_class,
                                            sizes[i : i + 1],
                                            coordinates[i : i + 1],
                                            self.size_p,
                                        )
                                    ),
                                    k=angle,
                                    axes=(3, 2),
                                )  # merge softmax scores from patches (overlaps)
                                scores[i] += np.rot90(
                                    np.array(
                                        patch2global(
                                            predicted_ensembles[i : i + 1],
                                            self.n_class,
                                            sizes[i : i + 1],
                                            coordinates[i : i + 1],
                                            self.size_p,
                                        )
                                    ),
                                    k=angle,
                                    axes=(3, 2),
                                )  # merge softmax scores from patches (overlaps)

            # global predictions 
            outputs_global = torch.Tensor(outputs_global)
            predictions_global = [
                F.interpolate(
                    outputs_global[i : i + 1], images[i].size[::-1], mode="nearest"
                )
                .argmax(1)
                .detach()
                .numpy()[0]
                for i in range(len(images))
            ]
            if not self.test:
                self.metrics_global.update(labels_npy, predictions_global)

            if self.mode is PhaseMode.LocalFromGlobal or self.mode is PhaseMode.GlobalFromLocal:
                # patch predictions 
                # predictions_local = scores_local.argmax(1) # b, h, w
                predictions_local = [score.argmax(1)[0] for score in scores_local]
                if not self.test:
                    self.metrics_local.update(labels_npy, predictions_local)
                
                # combined/ensemble predictions 
                # predictions = scores.argmax(1) # b, h, w
                predictions = [score.argmax(1)[0] for score in scores]
                if not self.test:
                    self.metrics.update(labels_npy, predictions)
                return predictions, predictions_global, predictions_local
            else:
                return None, predictions_global, None
