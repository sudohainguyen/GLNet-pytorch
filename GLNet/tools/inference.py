import os
import sys
import multiprocessing
import cv2
import numpy as np
from PIL import Image
from openslide import OpenSlide, deepzoom
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm

import torch
import torch.nn.functional as F

sys.path.append('../GLNet-pytorch/')
from GLNet.models.glnet import GLNet as glnet
from GLNet.utils.enums import PhaseMode
from GLNet.utils.processing import (
    images_transform,
    resize,
    global2patch,
    patch2global,
    add_extra_pixels
)
from GLNet.utils.filters import apply_filters
from GLNet.models.functional import crop_global
from GLNet.utils.visualization import apply_mask
from GLNet.options import InferenceOptions

DATA_DIR = '/home/hainq/quanghai/Thesis/datasets/paip2019/training/'

def load_models(n_class, path_l2g, path_g):
    model = glnet(n_class)
    # model = nn.DataParallel(model)
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
    # global_fixed = nn.DataParallel(global_fixed)
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


def predict_on_single_patch(image_as_tensor, model, global_fixed, size_g=(1008, 1008), size_p=(1008, 1008), n_class=2):
    # image_as_tensor[0] = apply_filters(image_as_tensor[0])
    images_global = resize(image_as_tensor, size_g)
    scores = [
        np.zeros((1, n_class, image_as_tensor[i].size[1], image_as_tensor[i].size[0]))
        for i in range(len(image_as_tensor))
    ]
    
    images_glb = images_transform(images_global)
    
    patches, coordinates, templates, sizes, ratios = global2patch(
        image_as_tensor, size_p
    )

    predicted_ensembles = [
        np.zeros((len(coordinates[i]),n_class,size_p[0],size_p[1]))
        for i in range(len(image_as_tensor))
    ]
    
    for i in range(len(image_as_tensor)):
        j = 0
        while j < len(coordinates[i]):
            patches_var = images_transform(
                patches[i][j : j + 1]
            )  # b, c, h, w

            fm_patches, _ = model.module.collect_local_fm(
                images_glb[i : i + 1],
                patches_var,
                ratios[i],
                coordinates[i],
                [j, j + 1],
                len(image_as_tensor),
                global_model=global_fixed,
                template=templates[i],
                n_patch_all=len(coordinates[i]),
            )
            j += 1
    
    _, fm_global = model.forward(
        images_glb, None, None, None, mode=PhaseMode.GlobalFromLocal
    )
    
    for i in range(len(image_as_tensor)):
        j = 0
        # while j < n ** 2:
        while j < len(coordinates[i]):
            fl = fm_patches[i][j : j + 1].cuda()
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
            predicted_ensembles[i][j : j + output_ensembles.size()[0]] += (
                F.interpolate(
                    output_ensembles,
                    size=size_p,
                    mode="nearest",
                )
                .data.cpu()
                .numpy()
            )
            j += 1
        
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

    predictions = [score.argmax(1)[0] for score in scores]
    return predictions


def predict_on_row(dz_obj, row, cols, model, global_fixed):
    for col in cols:
        tile = dz.get_tile(dz.level_count - 1, (col, row))
        if tile.size != (4096, 4096):
            tile = add_extra_pixels(tile, expected_shape=(4096, 4096))

def predict_on_svs(img_idx, model, global_fixed, save_pred=False):
    slide = OpenSlide(os.path.join(DATA_DIR, 'ws_images', f'{img_idx}.svs'))
    mask = io.imread(os.path.join(DATA_DIR, 'viable_masks', f'{img_idx}_viable.tif'))
    dz = deepzoom.DeepZoomGenerator(slide, tile_size=4096, overlap=0)
    img = slide.read_region((0, 0), 2, slide.level_dimensions[2])
    img = img.convert('RGB')
    img = np.array(img)
    
    cols, rows = dz.level_tiles[-1]
    
    num_processes = multiprocessing.cpu_count()
    if num_processes > cols:
        num_processes = cols

    # out = np.zeros_like(img)
    out = np.zeros((rows * 256, cols * 256, 3), dtype=np.uint8)
    
    for row in range(rows):
        for col in tqdm(range(cols)):
            tile = dz.get_tile(dz.level_count - 1, (col, row)) # col, row
            if tile.size != (4096, 4096):
                tile = add_extra_pixels(tile, expected_shape=(4096, 4096))
            pred = predict_on_single_patch([tile], model, global_fixed)
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
    return out


if __name__ == "__main__":
    args = InferenceOptions("Inference options").parse()
    model, global_fixed = load_models(args.n_class, args.path_l2g, args.path_g)
    predict_on_svs(args.img_idx, model, global_fixed, args.save_pred)
