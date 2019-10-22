import numpy as np
import torch

def apply_mask(np_img, mask, color=(0.2, 0.7, 0.7), alpha=0.5, inplace=False):
    """Apply the given mask to the image.
    """
    if inplace:
        image = np_img
    else:
        image = np_img.copy()
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def class_to_RGB(label):
    colmap = np.array([np.array(label)] * 3)
    colmap *= 255
    return torch.as_tensor(colmap, dtype=torch.uint8)
