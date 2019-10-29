from .resnet import get_resnet
from .resnet_dilation import get_dilated_resnet


def get_backbone(bb_name='resnet50', pretrained=False):
    if bb_name.startswith('resnet'):
        return get_resnet(bb_name=bb_name, pretrained=pretrained)
    if bb_name.startswith('d_resnet'):
        return get_dilated_resnet(bb_name=bb_name, pretrained=pretrained)
    if bb_name.startswith('vgg'):
        # TODO: implement VGGNet
        pass
    raise NotImplementedError(f'Your {bb_name} is not implemented yet or has invalid name')
