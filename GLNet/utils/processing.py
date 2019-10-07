import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

normalize = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

def resize(images, shape, is_mask=False):
    """
    resize PIL images
    shape: (w, h)
    """
    resized = list(images)
    for i, img in enumerate(images):
        if is_mask:
            resized[i] = img.resize(shape, Image.NEAREST)
        else:
            resized[i] = img.resize(shape, Image.BILINEAR).convert('RGB')
    return resized

def images_transform(images):
    """
    images: list of PIL images
    """
    inputs = []
    for img in images:
        inputs.append(normalize(img))
    inputs = torch.stack(inputs, dim=0).cuda()
    return inputs

def _mask_transform(mask):
    target = np.array(mask).astype("int32")
    # target[target == 255] = -1
    # target[target == 0] = -1
    # target -= 1 # in DeepGlobe: make class 0 (should be ignored) as -1 (to be ignored in cross_entropy)
    return target


def masks_transform(masks, numpy=False):
    """
    masks: list of PIL images
    """
    targets = []
    for m in masks:
        targets.append(_mask_transform(m))
    targets = np.array(targets)
    if numpy:
        return targets
    return torch.from_numpy(targets).long().cuda()

def get_patch_info(shape, p_size):
    """
    shape: origin image size, (x, y)
    p_size: patch size (square)
    return: n_x, n_y, step_x, step_y
    """
    x = shape[0]
    y = shape[1]
    n = m = 1
    while x > n * p_size:
        n += 1
    while p_size - 1.0 * (x - p_size) / (n - 1) < 50:
        n += 1
    while y > m * p_size:
        m += 1
    while p_size - 1.0 * (y - p_size) / (m - 1) < 50:
        m += 1
    return n, m, (x - p_size) * 1.0 / (n - 1), (y - p_size) * 1.0 / (m - 1)

def global2patch(images, p_size):
    """
    image/label => patches
    p_size: patch size
    return: list of PIL patch images; coordinates: images->patches; ratios: (h, w)
    """
    patches = []
    coordinates = []
    templates = []
    sizes = []
    ratios = [(0, 0)] * len(images)
    patch_ones = np.ones(p_size)

    for i, img in enumerate(images):
        w, h = img.size
        size = (h, w)
        sizes.append(size)
        ratios[i] = (float(p_size[0]) / size[0], float(p_size[1]) / size[1])
        template = np.zeros(size)
        n_x, n_y, step_x, step_y = get_patch_info(size, p_size[0])
        patches.append([img] * (n_x * n_y))
        coordinates.append([(0, 0)] * (n_x * n_y))
        for x in range(n_x):
            if x < n_x - 1:
                top = int(np.round(x * step_x))
            else:
                top = size[0] - p_size[0]

            for y in range(n_y):
                if y < n_y - 1:
                    left = int(np.round(y * step_y))
                else:
                    left = size[1] - p_size[1]

                template[top : top + p_size[0], left : left + p_size[1]] += patch_ones
                coordinates[i][x * n_y + y] = (
                    1.0 * top / size[0],
                    1.0 * left / size[1],
                )
                patches[i][x * n_y + y] = transforms.functional.crop(
                    img, top, left, p_size[0], p_size[1]
                )
        templates.append(Variable(torch.Tensor(template).expand(1, 1, -1, -1)).cuda())
    return patches, coordinates, templates, sizes, ratios


def patch2global(patches, n_class, sizes, coordinates, p_size):
    """
    predicted patches (after classify layer) => predictions
    return: list of np.array
    """
    predictions = [np.zeros((n_class, size[0], size[1])) for size in sizes]
    for i in range(len(sizes)):
        for j in range(len(coordinates[i])):
            top, left = coordinates[i][j]
            top = int(np.round(top * sizes[i][0]))
            left = int(np.round(left * sizes[i][1]))
            predictions[i][
                :, top : top + p_size[0], left : left + p_size[1]
            ] += patches[i][j]
    return predictions

def one_hot_gaussian_blur(index, classes):
    """
    index: numpy array b, h, w
    classes: int
    """
    mask = np.transpose(
        (np.arange(classes) == index[..., None]).astype(float), (0, 3, 1, 2)
    )
    b, c, _, _ = mask.shape
    for i in range(b):
        for j in range(c):
            mask[i][j] = cv2.GaussianBlur(mask[i][j], (0, 0), 8)

    return mask
