import torch
from torch import nn
from torch.nn import functional as F

from ..functional import upsample_add, concatenate

class FPN_Global(nn.Module):
    def __init__(self, num_class):
        super(FPN_Global, self).__init__()

        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        # Classify layers
        self.classify = nn.Conv2d(128 * 4, num_class, kernel_size=3, stride=1, padding=1)

        # Declare layers which handle channels from local
        # Top layer
        self.toplayer_ext = nn.Conv2d(2048 * 2, 256, kernel_size=1, stride=1, padding=0)
        # Lateral layers
        self.latlayer1_ext = nn.Conv2d(1024 * 2, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2_ext = nn.Conv2d(512 * 2, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3_ext = nn.Conv2d(256 * 2, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1_1_ext = nn.Conv2d(256 * 2, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_1_ext = nn.Conv2d(256 * 2, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_1_ext = nn.Conv2d(256 * 2, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_1_ext = nn.Conv2d(256 * 2, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1_2_ext = nn.Conv2d(256 * 2, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_2_ext = nn.Conv2d(256 * 2, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_2_ext = nn.Conv2d(256 * 2, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4_2_ext = nn.Conv2d(256 * 2, 128, kernel_size=3, stride=1, padding=1)
        self.smooth = nn.Conv2d(128 * 4 * 2, 128 * 4, kernel_size=3, stride=1, padding=1)

    def forward(self, backbone_outputs, additional_fms=None, ps_exts=None):
        c2, c3, c4, c5 = backbone_outputs
        
        if additional_fms:
            c2_ext, c3_ext, c4_ext, c5_ext = additional_fms

        if ps_exts:
            ps0_ext, ps1_ext, ps2_ext = ps_exts
        
        # Top-down
        if c5_ext is None:
            p5 = self.toplayer(c5)
            p4 = upsample_add(p5, self.latlayer1(c4))
            p3 = upsample_add(p4, self.latlayer2(c3))
            p2 = upsample_add(p3, self.latlayer3(c2))
        else:
            p5 = self.toplayer_ext(torch.cat((c5, c5_ext), dim=1))
            p4 = upsample_add(
                p5, self.latlayer1_ext(torch.cat((c4, c4_ext), dim=1))
            )
            p3 = upsample_add(
                p4, self.latlayer2_ext(torch.cat((c3, c3_ext), dim=1))
            )
            p2 = upsample_add(
                p3, self.latlayer3_ext(torch.cat((c2, c2_ext), dim=1))
            )
        ps0 = [p5, p4, p3, p2]

        # Smooth
        if ps0_ext is None:
            p5 = self.smooth1_1(p5)
            p4 = self.smooth2_1(p4)
            p3 = self.smooth3_1(p3)
            p2 = self.smooth4_1(p2)
        else:
            p5 = self.smooth1_1_ext(torch.cat((p5, ps0_ext[0]), dim=1))
            p4 = self.smooth2_1_ext(torch.cat((p4, ps0_ext[1]), dim=1))
            p3 = self.smooth3_1_ext(torch.cat((p3, ps0_ext[2]), dim=1))
            p2 = self.smooth4_1_ext(torch.cat((p2, ps0_ext[3]), dim=1))
        ps1 = [p5, p4, p3, p2]

        if ps1_ext is None:
            p5 = self.smooth1_2(p5)
            p4 = self.smooth2_2(p4)
            p3 = self.smooth3_2(p3)
            p2 = self.smooth4_2(p2)
        else:
            p5 = self.smooth1_2_ext(torch.cat((p5, ps1_ext[0]), dim=1))
            p4 = self.smooth2_2_ext(torch.cat((p4, ps1_ext[1]), dim=1))
            p3 = self.smooth3_2_ext(torch.cat((p3, ps1_ext[2]), dim=1))
            p2 = self.smooth4_2_ext(torch.cat((p2, ps1_ext[3]), dim=1))
        ps2 = [p5, p4, p3, p2]

        # Classify
        if ps2_ext is None:
            ps3 = concatenate(p5, p4, p3, p2)
            output = self.classify(ps3)
        else:
            p = concatenate(
                torch.cat((p5, ps2_ext[0]), dim=1),
                torch.cat((p4, ps2_ext[1]), dim=1),
                torch.cat((p3, ps2_ext[2]), dim=1),
                torch.cat((p2, ps2_ext[3]), dim=1),
            )
            ps3 = self.smooth(p)
            output = self.classify(ps3)

        return output, ps0, ps1, ps2, ps3

class FPN_Local(nn.Module):
    def __init__(self, num_class):
        super(FPN_Local, self).__init__()
        # Top layer
        fold = 2
        self.toplayer = nn.Conv2d(2048 * fold, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Lateral layers [C]
        self.latlayer1 = nn.Conv2d(1024 * fold, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512 * fold, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256 * fold, 256, kernel_size=1, stride=1, padding=0)
        
        # Smooth layers
        # ps0
        self.smooth1_1 = nn.Conv2d(256 * fold, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(256 * fold, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(256 * fold, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(256 * fold, 256, kernel_size=3, stride=1, padding=1)
        # ps1
        self.smooth1_2 = nn.Conv2d(256 * fold, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(256 * fold, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(256 * fold, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(256 * fold, 128, kernel_size=3, stride=1, padding=1)
        # ps2 is concatenation
        # Classify layers
        self.smooth = nn.Conv2d(128 * 4 * fold, 128 * 4, kernel_size=3, stride=1, padding=1)
        self.classify = nn.Conv2d(128 * 4, num_class, kernel_size=3, stride=1, padding=1)

    def forward(self, backbone_outputs, additional_fms=None, ps_exts=None):
        c2, c3, c4, c5 = backbone_outputs
        
        if additional_fms:
            c2_ext, c3_ext, c4_ext, c5_ext = additional_fms

        if ps_exts:
            ps0_ext, ps1_ext, ps2_ext = ps_exts
        
        # Top-down
        p5 = self.toplayer(
            torch.cat(
                [c5] + [F.interpolate(c5_ext[0], size=c5.size()[2:], **self._up_kwargs)],
                dim=1,
            )
        )
        p4 = self._upsample_add(
            p5,
            self.latlayer1(
                torch.cat(
                    [c4] + [F.interpolate(c4_ext[0], size=c4.size()[2:], **self._up_kwargs)],
                    dim=1,
                )
            ),
        )
        p3 = self._upsample_add(
            p4,
            self.latlayer2(
                torch.cat(
                    [c3] + [F.interpolate(c3_ext[0], size=c3.size()[2:], **self._up_kwargs)],
                    dim=1,
                )
            ),
        )
        p2 = self._upsample_add(
            p3,
            self.latlayer3(
                torch.cat(
                    [c2] + [F.interpolate(c2_ext[0], size=c2.size()[2:], **self._up_kwargs)],
                    dim=1,
                )
            ),
        )
        ps0 = [p5, p4, p3, p2]

        # Smooth
        p5 = self.smooth1_1(
            torch.cat(
                [p5] + [F.interpolate(ps0_ext[0][0], size=p5.size()[2:], **self._up_kwargs)],
                dim=1,
            )
        )
        p4 = self.smooth2_1(
            torch.cat(
                [p4] + [F.interpolate(ps0_ext[1][0], size=p4.size()[2:], **self._up_kwargs)],
                dim=1,
            )
        )
        p3 = self.smooth3_1(
            torch.cat(
                [p3] + [F.interpolate(ps0_ext[2][0], size=p3.size()[2:], **self._up_kwargs)],
                dim=1,
            )
        )
        p2 = self.smooth4_1(
            torch.cat(
                [p2] + [F.interpolate(ps0_ext[3][0], size=p2.size()[2:], **self._up_kwargs)],
                dim=1,
            )
        )
        ps1 = [p5, p4, p3, p2]

        p5 = self.smooth1_2(
            torch.cat(
                [p5] + [F.interpolate(ps1_ext[0][0], size=p5.size()[2:], **self._up_kwargs)],
                dim=1,
            )
        )
        p4 = self.smooth2_2(
            torch.cat(
                [p4] + [F.interpolate(ps1_ext[1][0], size=p4.size()[2:], **self._up_kwargs)],
                dim=1,
            )
        )
        p3 = self.smooth3_2(
            torch.cat(
                [p3] + [F.interpolate(ps1_ext[2][0], size=p3.size()[2:], **self._up_kwargs)],
                dim=1,
            )
        )
        p2 = self.smooth4_2(
            torch.cat(
                [p2] + [F.interpolate(ps1_ext[3][0], size=p2.size()[2:], **self._up_kwargs)],
                dim=1,
            )
        )
        ps2 = [p5, p4, p3, p2]

        # Classify
        # use ps2_ext
        ps3 = self._concatenate(
            torch.cat(
                [p5] + [F.interpolate(ps2_ext[0][0], size=p5.size()[2:], **self._up_kwargs)],
                dim=1,
            ),
            torch.cat(
                [p4] + [F.interpolate(ps2_ext[1][0], size=p4.size()[2:], **self._up_kwargs)],
                dim=1,
            ),
            torch.cat(
                [p3] + [F.interpolate(ps2_ext[2][0], size=p3.size()[2:], **self._up_kwargs)],
                dim=1,
            ),
            torch.cat(
                [p2] + [F.interpolate(ps2_ext[3][0], size=p2.size()[2:], **self._up_kwargs)],
                dim=1,
            ),
        )
        ps3 = self.smooth(ps3)
        output = self.classify(ps3)

        return output, ps0, ps1, ps2, ps3
