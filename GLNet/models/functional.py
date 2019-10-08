import torch
from torch.nn import functional as F
import numpy as np

__all__ = ['concatenate', 'upsample_add']

kwargs = {"mode": "bilinear", "align_corners": True}

def concatenate(p5, p4, p3, p2):
    _, _, H, W = p2.size()
    p5 = F.interpolate(p5, size=(H, W), **kwargs)
    p4 = F.interpolate(p4, size=(H, W), **kwargs)
    p3 = F.interpolate(p3, size=(H, W), **kwargs)
    return torch.cat([p5, p4, p3, p2], dim=1)


def upsample_add(x, y):
    """Upsample and add two feature maps.
    Args:
        x: (Variable) top feature map to be upsampled.
        y: (Variable) lateral feature map.
    Returns:
        (Variable) added feature map.
    Note in PyTorch, when input size is odd, the upsampled feature map
    with `F.interpolate(..., scale_factor=2, mode='nearest')`
    maybe not equal to the lateral feature map size.
    e.g.
    original input size: [N,_,15,15] ->
    conv2d feature map size: [N,_,8,8] ->
    upsampled feature map size: [N,_,16,16]
    So we choose bilinear upsample which supports arbitrary output sizes.
    """
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), **kwargs) + y

def sample_grid(fm, bbox, sample_size):
    """
    :param fm: tensor(b,c,h,w) the global feature map
    :param bbox: list [b* nparray(x1, y1, x2, y2)] the (x1,y1) is the left_top of bbox, (x2, y2) is the right_bottom of bbox
    there are in range [0, 1]. x is corresponding to width dimension and y is corresponding to height dimension
    :param sample_size: (oH, oW) the point to sample in height dimension and width dimension
    :return: tensor(b, c, oH, oW) sampled tensor
    """
    b = fm.shape[0]
    b_bbox = len(bbox)
    bbox = [x * 2 - 1 for x in bbox]  # range transform
    if b != b_bbox and b == 1:
        fm = torch.cat([fm] * b_bbox, dim=0)
    grid = np.zeros((b_bbox,) + sample_size + (2,), dtype=np.float32)
    gridMap = np.array(
        [
            [
                (cnt_w / (sample_size[1] - 1), cnt_h / (sample_size[0] - 1))
                for cnt_w in range(sample_size[1])
            ]
            for cnt_h in range(sample_size[0])
        ]
    )
    for cnt_b in range(b_bbox):
        grid[cnt_b, :, :, 0] = (
            bbox[cnt_b][0] + (bbox[cnt_b][2] - bbox[cnt_b][0]) * gridMap[:, :, 0]
        )
        grid[cnt_b, :, :, 1] = (
            bbox[cnt_b][1] + (bbox[cnt_b][3] - bbox[cnt_b][1]) * gridMap[:, :, 1]
        )
    grid = torch.from_numpy(grid).cuda()
    return F.grid_sample(fm, grid)

def crop_global(f_global, top_lefts, ratio):
    """
    top_lefts: [(top, left)] * b
    """
    _, _, H, W = f_global.size()
    b = len(top_lefts)
    h, w = int(np.round(H * ratio[0])), int(np.round(W * ratio[1]))

    # bbox = [ np.array([left, top, left + ratio, top + ratio]) for (top, left) in top_lefts ]
    # crop = self._sample_grid(f_global, bbox, (H, W))

    crop = []
    for i in range(b):
        top, left = (
            int(np.round(top_lefts[i][0] * H)),
            int(np.round(top_lefts[i][1] * W)),
        )
        # # global's sub-region & upsample
        # f_global_patch = F.interpolate(f_global[0:1, :, top:top+h, left:left+w], size=(h, w), mode='bilinear')
        f_global_patch = f_global[0:1, :, top : top + h, left : left + w]
        crop.append(f_global_patch[0])
    crop = torch.stack(crop, dim=0)  # stack into mini-batch
    return [crop]  # return as a list for easy to torch.cat

def merge_local(f_local, merge, f_global, top_lefts, oped, ratio, template, up_kwargs):
    """
    merge feature maps from local patches, and finally to a whole image's feature map (on cuda)
    f_local: a sub_batch_size of patch's feature map
    oped: [start, end)
    """
    b = f_local.size()[0]
    _, c, H, W = f_global.size()  # match global feature size
    if merge is None:
        merge = torch.zeros((1, c, H, W)).cuda()
    h, w = int(np.round(H * ratio[0])), int(np.round(W * ratio[1]))
    for i in range(b):
        index = oped[0] + i
        top, left = (
            int(np.round(H * top_lefts[index][0])),
            int(np.round(W * top_lefts[index][1])),
        )
        merge[:, :, top : top + h, left : left + w] += F.interpolate(
            f_local[i : i + 1], size=(h, w), **up_kwargs
        )
    if oped[1] >= len(top_lefts):
        template = F.interpolate(template, size=(H, W), **up_kwargs)
        template = template.expand_as(merge)
        # template = Variable(template).cuda()
        merge /= template
    return merge
