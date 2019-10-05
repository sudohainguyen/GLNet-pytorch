import os

import random
import numpy as np
import cv2
from PIL import Image, ImageFile

from torch.utils import data
from torchvision import transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

class PAIP(data.Dataset):

    def __init__(self, root, ids, labels=False, transforms=False):
        super(PAIP, self).__init__()
        self.root = root
        self.labels = labels
        self.transforms = transforms
        self.ids = ids
        self.classdict = {
            1: 'yes'
        }

        self.resizer = transforms.Resize((4096, 4096))

    def __getitem__(self, index):
        sample = {'id': self.ids[index][:-8]}
        image = Image.open(os.path.join(self.root, ''))
