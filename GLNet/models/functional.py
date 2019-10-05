import torch
from torch.nn import functional as F

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
