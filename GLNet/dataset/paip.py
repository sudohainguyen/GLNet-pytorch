import os

import random
import numpy as np
from PIL import Image, ImageFile

from torch.utils import data
from torchvision import transforms
from skimage import io

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])


def _transform(image, label):
    if np.random.random() > 0.5:
        image = transforms.functional.hflip(image)
        label = transforms.functional.hflip(label)

    if np.random.random() > 0.5:
        degree = random.choice([90, 180, 270])
        image = transforms.functional.rotate(image, degree)
        label = transforms.functional.rotate(label, degree)
    return image, label


def add_extra_pixels(image, expected_shape=(2048, 2048), is_mask=False):
    if is_mask:
        temp = np.zeros(expected_shape, dtype=np.uint8)
    else:
        image = np.array(image)
        temp = np.zeros((expected_shape[0], expected_shape[1], 3), dtype=np.uint8)
        temp += 243
    
    temp[:image.shape[0], :image.shape[1]] = image
    return Image.fromarray(temp)


class Paip(data.Dataset):

    def __init__(self, root, ids, label=False, img_shape=2048, transform=False):
        super(Paip, self).__init__()
        self.root = root
        self.label = label
        self.transform = transform
        self.ids = ids
        self.img_shape = (img_shape, img_shape)
        # self.classdict = {
        #     0: 'no',
        #     1: 'yes'
        # }

        # self.resizer = transforms.Resize((4096, 4096))

    def __getitem__(self, index):
        sample = {'id': self.ids[index][:14]}
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
    