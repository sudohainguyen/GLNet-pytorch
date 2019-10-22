import os

import random
import numpy as np
from PIL import Image, ImageFile

import torch
from torch.utils import data
from torchvision import transforms
from skimage import io

from ..utils.filters import (
    filter_green_channel,
    filter_red_pen,
    filter_blue_pen,
    filter_green_pen,
    filter_remove_small_objects,
    filter_grays,
    mask_rgb,
    add_extra_pixels,
    _transform
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])


def class_to_RGB(label):
    colmap = np.array([np.array(label)] * 3)
    colmap *= 255
    return torch.as_tensor(colmap, dtype=torch.uint8)


def apply_filters(np_img):
    """
    Apply filters to image as NumPy array and optionally save and/or display filtered images.
    Args:
        np_img: Image as NumPy array.
    Returns:
        Resulting filtered image as a NumPy array.
    """
    rgb = np_img

    mask_not_green = filter_green_channel(rgb)
    # rgb_not_green = mask_rgb(rgb, mask_not_green)

    mask_not_gray = filter_grays(rgb)
    # rgb_not_gray = mask_rgb(rgb, mask_not_gray)

    mask_no_red_pen = filter_red_pen(rgb)
    # rgb_no_red_pen = mask_rgb(rgb, mask_no_red_pen)

    mask_no_green_pen = filter_green_pen(rgb)
    # rgb_no_green_pen = mask_rgb(rgb, mask_no_green_pen)

    mask_no_blue_pen = filter_blue_pen(rgb)
    # rgb_no_blue_pen = mask_rgb(rgb, mask_no_blue_pen)

    mask_gray_green_pens = (
        mask_not_gray
        & mask_not_green
        & mask_no_red_pen
        & mask_no_green_pen
        & mask_no_blue_pen
    )
    # rgb_gray_green_pens = mask_rgb(rgb, mask_gray_green_pens)

    mask_remove_small = filter_remove_small_objects(
        mask_gray_green_pens, min_size=500, output_type="bool"
    )
    rgb_remove_small = mask_rgb(rgb, mask_remove_small)

    img = rgb_remove_small
    return img


class Paip(data.Dataset):

    def __init__(self, root, ids, label=False, img_shape=2048, transform=False):
        super(Paip, self).__init__()
        self.root = root
        self.label = label
        self.transform = transform
        self.ids = ids
        self.img_shape = (img_shape, img_shape)

        # self.resizer = transforms.Resize((4096, 4096))

    def __getitem__(self, index):
        sample = {'id': self.ids[index][:-4]}
        image = Image.open(os.path.join(self.root, 'images', self.ids[index]))

        # handle edge patches
        if image.size != self.img_shape:
            image = add_extra_pixels(image, expected_shape=self.img_shape)

        if self.label:
            label = io.imread(os.path.join(self.root, 'masks', self.ids[index].replace('.jpg', '_mask.tif')))
            if label.size != self.img_shape:
                label = add_extra_pixels(label, expected_shape=self.img_shape, is_mask=True)

        if self.transform and self.label:
            image, label = _transform(image, label)

        sample['image'] = image
        sample['label'] = label
        return sample

    def __len__(self):
        return len(self.ids)
    