import os
import sys
import cv2
import numpy as np
from PIL import Image
from openslide import OpenSlide, deepzoom
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append('../GLNet-pytorch/')
from GLNet.models.glnet import GLNet as glnet
from GLNet.helpers.evaluator import Evaluator
from GLNet.utils.enums import PhaseMode
from GLNet.utils.processing import (
    images_transform,
    resize,
    global2patch,
    patch2global
)
from GLNet.models.functional import crop_global
from GLNet.utils.visualization import apply_mask

DATA_DIR = '../../datasets/paip2019/training/'

def load_models(n_class, path_l2g, path_g):
    model = glnet(n_class)
    model = nn.DataParallel(model)
    model.cuda()
    partial = torch.load(path_l2g)
    # partial = torch.load('../../checkpoints/fpn_mode2_paip.1008_lr1e4.pth')
    state = model.state_dict()
    pretrained_dict = {
        k: v for k, v in partial.items() if k in state
    }
    state.update(pretrained_dict)
    model.load_state_dict(state)
    model.eval()

    global_fixed = glnet(2)
    global_fixed = nn.DataParallel(global_fixed)
    global_fixed.cuda()
    partial = torch.load(path_g)
    state = global_fixed.state_dict()
    pretrained_dict = {
        k: v for k, v in partial.items() if k in state and "local" not in k
    }
    # 2. overwrite entries in the existing state dict
    state.update(pretrained_dict)
    # 3. load the new state dict
    global_fixed.load_state_dict(state)
    global_fixed.eval()

    return model, global_fixed


def predict_on_patch(images, model, global_fixed, size_g=(1008, 1008), size_p=(1008, 1008), n_class=2):
    images_global = resize(images, size_g)
    outputs_global = np.zeros(
        (len(images), n_class, size_g[0] // 4, size_g[1] // 4)
    )
    # images_local = [image.copy() for image in images]

    scores_local = [
        np.zeros((1, n_class, images[i].size[1], images[i].size[0]))
        for i in range(len(images))
    ]
    scores = [
        np.zeros((1, n_class, images[i].size[1], images[i].size[0]))
        for i in range(len(images))
    ]
    
    images_glb = images_transform(images_global)
    
    patches, coordinates, templates, sizes, ratios = global2patch(
        images, size_p
    )

    predicted_patches = [
        np.zeros((len(coordinates[i]),n_class,size_p[0],size_p[1]))
        for i in range(len(images))
    ]
    predicted_ensembles = [
        np.zeros((len(coordinates[i]),n_class,size_p[0],size_p[1]))
        for i in range(len(images))
    ]
    
    for i in range(len(images)):
        j = 0
        while j < len(coordinates[i]):
            patches_var = images_transform(
                patches[i][j : j + 1]
            )  # b, c, h, w

            fm_patches, output_patches = model.module.collect_local_fm(
                images_glb[i : i + 1],
                patches_var,
                ratios[i],
                coordinates[i],
                [j, j + 1],
                len(images),
                global_model=global_fixed,
                template=templates[i],
                n_patch_all=len(coordinates[i]),
            )
            predicted_patches[i][
                j : j + output_patches.size()[0]
            ] += (
                F.interpolate(
                    output_patches, size=size_p, mode="nearest"
                )
                .data.cpu()
                .numpy()
            )
            j += 1
    
    tmp, fm_global = model.forward(
        images_glb, None, None, None, mode=PhaseMode.GlobalFromLocal
    )
    outputs_global += np.rot90(tmp.data.cpu().numpy(), k=0, axes=(3, 2))
    
    for i in range(len(images)):
        j = 0
        # while j < n ** 2:
        while j < len(coordinates[i]):
            fl = fm_patches[i][j : j + 1].cuda()
            # fg = model.module._crop_global(fm_global[i:i+1], coordinates[j:j+1], ratio)[0]
            fg = crop_global(
                fm_global[i : i + 1],
                coordinates[i][j : j + 1],
                ratios[i])[0]
            fg = F.interpolate(
                fg, size=fl.size()[2:], mode="bilinear"
            )
            output_ensembles = model.module.ensemble(
                fl, fg
            )  # include cordinates
            # output_ensembles = F.interpolate(model.module.ensemble(fl, fg), size_p, **model.module._up_kwargs)

            # ensemble predictions
            predicted_ensembles[i][
                j : j + output_ensembles.size()[0]
            ] += (
                F.interpolate(
                    output_ensembles,
                    size=size_p,
                    mode="nearest",
                )
                .data.cpu()
                .numpy()
            )
            j += 1
        scores_local[i] += np.rot90(
            np.array(
                patch2global(
                    predicted_patches[i : i + 1],
                    n_class,
                    sizes[i : i + 1],
                    coordinates[i : i + 1],
                    size_p,
                )
            ),
            k=0,
            axes=(3, 2),
        )  # merge softmax scores from patches (overlaps)
        scores[i] += np.rot90(
            np.array(
                patch2global(
                    predicted_ensembles[i : i + 1],
                    n_class,
                    sizes[i : i + 1],
                    coordinates[i : i + 1],
                    size_p,
                )
            ),
            k=0,
            axes=(3, 2),
        )
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
    predictions_local = [score.argmax(1)[0] for score in scores_local]
    predictions = [score.argmax(1)[0] for score in scores]
    
    return predictions, predictions_global, predictions_local


def predict_on_svs(img_idx, model, global_fixed, save_pred=False):
    slide = OpenSlide(os.path.join(DATA_DIR, 'ws_images', f'{img_idx}.svs'))
    mask = io.imread(os.path.join(DATA_DIR, 'viable_masks', f'{img_idx}_viable.tif'))
    dz = deepzoom.DeepZoomGenerator(slide, tile_size=4096, overlap=0)
    img = slide.read_region((0, 0), 2, slide.level_dimensions[2])
    img = img.convert('RGB')
    img = np.array(img)
    
    out = np.zeros_like(img)
    
    cols, rows = dz.level_tiles[-1]
    for row in range(rows):
        for col in tqdm(range(cols)):
            tile = dz.get_tile(dz.level_count - 1, (col, row)) # col, row
            if tile.size != (4096, 4096):
                continue
            pred, _, _ = predict_on_patch([tile], model, global_fixed)
            pil_pred = Image.fromarray(pred[0].astype(np.uint8))

            pil_pred = pil_pred.resize((pil_pred.size[0] // 16, pil_pred.size[1] // 16), Image.NEAREST)
            tile = tile.resize((tile.size[0] // 16, tile.size[1] // 16), Image.BILINEAR).convert('RGB')

            pil_pred = np.array(pil_pred)
            tile = np.array(tile)

            applied_mask = apply_mask(tile, pil_pred)
    #         plt.imshow(applied_mask)

            edge_right = (col + 1) * tile.shape[0]
            edge_bot = (row + 1) * tile.shape[1]
    #         if edge_right > out.shape[0]:
    #             edge_right = out.shape[0]
    #         if edge_bot > out.shape[1]:
    #             edge_bot = out.shape[1]

            out[row * tile.shape[1] : edge_bot, \
                col * tile.shape[0] : edge_right, :] = applied_mask
    
    mask = cv2.resize(mask, (mask.shape[1] // 16, mask.shape[0] // 16))
    applied_mask = apply_mask(np.array(img), mask)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.axis('off')
    ax1.imshow(applied_mask)
    ax1.set_title('Ground Truth', fontsize=16)
    ax2.axis('off')
    ax2.imshow(out)
    ax2.set_title('Prediction', fontsize=16)
    if save_pred:
        fig.savefig(f'Pred_{img_idx}.png')
