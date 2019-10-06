from torch import nn
from torch.nn import functional as F
import torch

from .backbones.resnet import resnet50
from .extractors.fpn import FPN_Global, FPN_Local
from .functional import crop_global, merge_local
from ..utils import PhaseMode

class GLNet(nn.Module):
    def __init__(self, num_class):
        super(GLNet, self).__init__()

        self._up_kwargs = {"mode": "bilinear"}

        self.backbone_global = resnet50(pretrained=True)
        self.backbone_local = resnet50(pretrained=True)

        self.fpn_global = FPN_Global(num_class)
        self.fpn_local = FPN_Local(num_class)

        self.c2_g = None
        self.c3_g = None
        self.c4_g = None
        self.c5_g = None
        self.output_g = None

        self.ps0_g = None
        self.ps1_g = None
        self.ps2_g = None
        self.ps3_g = None

        self.c2_l = []
        self.c3_l = []
        self.c4_l = []
        self.c5_l = []

        self.ps00_l = []
        self.ps01_l = []
        self.ps02_l = []
        self.ps03_l = []

        self.ps10_l = []
        self.ps11_l = []
        self.ps12_l = []
        self.ps13_l = []

        self.ps20_l = []
        self.ps21_l = []
        self.ps22_l = []
        self.ps23_l = []

        self.ps0_l = None
        self.ps1_l = None
        self.ps2_l = None
        self.ps3_l = []  # self.output_l = []

        self.c2_b = None
        self.c3_b = None
        self.c4_b = None
        self.c5_b = None

        self.ps00_b = None
        self.ps01_b = None
        self.ps02_b = None
        self.ps03_b = None

        self.ps10_b = None
        self.ps11_b = None
        self.ps12_b = None
        self.ps13_b = None

        self.ps20_b = None
        self.ps21_b = None
        self.ps22_b = None
        self.ps23_b = None
        self.ps3_b = []  # self.output_b = []

        self.patch_n = 0
        
        self.ensemble_conv = nn.Conv2d(128 * 4 * 2, num_class, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.ensemble_conv.weight, mean=0, std=0.01)

        # init fpns
        for m in self.fpn_global.children():
            if hasattr(m, "weight"):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, "bias"):
                nn.init.constant_(m.bias, 0)
        for m in self.fpn_local.children():
            if hasattr(m, "weight"):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            if hasattr(m, "bias"):
                nn.init.constant_(m.bias, 0)
    
    def clear_cache(self):
        self.c2_g = None
        self.c3_g = None
        self.c4_g = None
        self.c5_g = None
        self.output_g = None

        self.ps0_g = None
        self.ps1_g = None
        self.ps2_g = None
        self.ps3_g = None

        self.c2_l = []
        self.c3_l = []
        self.c4_l = []
        self.c5_l = []

        self.ps00_l = []
        self.ps01_l = []
        self.ps02_l = []
        self.ps03_l = []

        self.ps10_l = []
        self.ps11_l = []
        self.ps12_l = []
        self.ps13_l = []

        self.ps20_l = []
        self.ps21_l = []
        self.ps22_l = []
        self.ps23_l = []

        self.ps0_l = None
        self.ps1_l = None
        self.ps2_l = None
        self.ps3_l = []
        # self.output_l = []

        self.c2_b = None
        self.c3_b = None
        self.c4_b = None
        self.c5_b = None

        self.ps00_b = None
        self.ps01_b = None
        self.ps02_b = None
        self.ps03_b = None

        self.ps10_b = None
        self.ps11_b = None
        self.ps12_b = None
        self.ps13_b = None

        self.ps20_b = None
        self.ps21_b = None
        self.ps22_b = None
        self.ps23_b = None

        self.ps3_b = []
        # self.output_b = []

        self.patch_n = 0

    def ensemble(self, f_local, f_global):
        return self.ensemble_conv(torch.cat((f_local, f_global), dim=1))

    def collect_local_fm(
        self,
        image_global,
        patches,
        ratio,
        top_lefts,
        oped,
        batch_size,
        global_model=None,
        template=None,
        n_patch_all=None,
    ):
        """
        patches: 1 patch
        top_lefts: all top-left
        oped: [start, end)
        """
        with torch.no_grad():
            if self.patch_n == 0:
                self.c2_g, self.c3_g, self.c4_g, self.c5_g = \
                    global_model.module.backbone_global.forward(image_global)

                self.output_g, self.ps0_g, self.ps1_g, self.ps2_g, self.ps3_g = \
                    global_model.module.fpn_global.forward(self.c2_g, self.c3_g, self.c4_g, self.c5_g)

                # self.output_g = F.interpolate(self.output_g, image_global.size()[2:], mode='nearest')
            self.patch_n += patches.size()[0]
            self.patch_n %= n_patch_all

            self.backbone_local.eval()
            self.fpn_local.eval()

            (c2, c3, c4, c5) = self.backbone_local.forward(patches)

            c2_ext = crop_global(self.c2_g, top_lefts[oped[0] : oped[1]], ratio)
            c3_ext = crop_global(self.c3_g, top_lefts[oped[0] : oped[1]], ratio)
            c4_ext = crop_global(self.c4_g, top_lefts[oped[0] : oped[1]], ratio)
            c5_ext = crop_global(self.c5_g, top_lefts[oped[0] : oped[1]], ratio)

            ps0_ext = [crop_global(f, top_lefts[oped[0] : oped[1]], ratio) for f in self.ps0_g]
            ps1_ext = [crop_global(f, top_lefts[oped[0] : oped[1]], ratio) for f in self.ps1_g]
            ps2_ext = [crop_global(f, top_lefts[oped[0] : oped[1]], ratio) for f in self.ps2_g]

            # global's 1x patch cat
            output, ps0, ps1, ps2, ps3 = self.fpn_local.forward(
                backbone_outputs=(c2, c3, c4, c5),
                additional_fms=(c2_ext, c3_ext, c4_ext, c5_ext),
                ps_exts=(ps0_ext, ps1_ext, ps2_ext)
            )
            # output = F.interpolate(output, patches.size()[2:], mode='nearest')

            self.c2_b = merge_local(c2, self.c2_b, self.c2_g, top_lefts, oped, ratio, template, self._up_kwargs)
            self.c3_b = merge_local(c3, self.c3_b, self.c3_g, top_lefts, oped, ratio, template, self._up_kwargs)
            self.c4_b = merge_local(c4, self.c4_b, self.c4_g, top_lefts, oped, ratio, template, self._up_kwargs)
            self.c5_b = merge_local(c5, self.c5_b, self.c5_g, top_lefts, oped, ratio, template, self._up_kwargs)

            self.ps00_b = merge_local(ps0[0], self.ps00_b, self.ps0_g[0], top_lefts, oped, ratio, template, self._up_kwargs)
            self.ps01_b = merge_local(ps0[1], self.ps01_b, self.ps0_g[1], top_lefts, oped, ratio, template, self._up_kwargs)
            self.ps02_b = merge_local(ps0[2], self.ps02_b, self.ps0_g[2], top_lefts, oped, ratio, template, self._up_kwargs)
            self.ps03_b = merge_local(ps0[3], self.ps03_b, self.ps0_g[3], top_lefts, oped, ratio, template, self._up_kwargs)
            self.ps10_b = merge_local(ps1[0], self.ps10_b, self.ps1_g[0], top_lefts, oped, ratio, template, self._up_kwargs)
            self.ps11_b = merge_local(ps1[1], self.ps11_b, self.ps1_g[1], top_lefts, oped, ratio, template, self._up_kwargs)
            self.ps12_b = merge_local(ps1[2], self.ps12_b, self.ps1_g[2], top_lefts, oped, ratio, template, self._up_kwargs)
            self.ps13_b = merge_local(ps1[3], self.ps13_b, self.ps1_g[3], top_lefts, oped, ratio, template, self._up_kwargs)
            self.ps20_b = merge_local(ps2[0], self.ps20_b, self.ps2_g[0], top_lefts, oped, ratio, template, self._up_kwargs)
            self.ps21_b = merge_local(ps2[1], self.ps21_b, self.ps2_g[1], top_lefts, oped, ratio, template, self._up_kwargs)
            self.ps22_b = merge_local(ps2[2], self.ps22_b, self.ps2_g[2], top_lefts, oped, ratio, template, self._up_kwargs)
            self.ps23_b = merge_local(ps2[3], self.ps23_b, self.ps2_g[3], top_lefts, oped, ratio, template, self._up_kwargs)

            self.ps3_b.append(ps3.cpu())
            # self.output_b.append(output.cpu()) # each output is 1, 7, h, w

            if self.patch_n == 0:
                # merged all patches into an image
                self.c2_l.append(self.c2_b)
                self.c3_l.append(self.c3_b)
                self.c4_l.append(self.c4_b)
                self.c5_l.append(self.c5_b)

                self.ps00_l.append(self.ps00_b)
                self.ps01_l.append(self.ps01_b)
                self.ps02_l.append(self.ps02_b)
                self.ps03_l.append(self.ps03_b)

                self.ps10_l.append(self.ps10_b)
                self.ps11_l.append(self.ps11_b)
                self.ps12_l.append(self.ps12_b)
                self.ps13_l.append(self.ps13_b)

                self.ps20_l.append(self.ps20_b)
                self.ps21_l.append(self.ps21_b)
                self.ps22_l.append(self.ps22_b)
                self.ps23_l.append(self.ps23_b)

                # collected all ps3 and output of patches as a (b) tensor, append into list
                self.ps3_l.append(torch.cat(self.ps3_b, dim=0))  # a list of tensors
                # self.output_l.append(torch.cat(self.output_b, dim=0)) # a list of 36, 7, h, w tensors

                self.c2_b = None
                self.c3_b = None
                self.c4_b = None
                self.c5_b = None

                self.ps00_b = None
                self.ps01_b = None
                self.ps02_b = None
                self.ps03_b = None

                self.ps10_b = None
                self.ps11_b = None
                self.ps12_b = None
                self.ps13_b = None

                self.ps20_b = None
                self.ps21_b = None
                self.ps22_b = None
                self.ps23_b = None
                self.ps3_b = []  # ; self.output_b = []
            if len(self.c2_l) == batch_size:
                self.c2_l = torch.cat(self.c2_l, dim=0)  # .cuda()
                self.c3_l = torch.cat(self.c3_l, dim=0)  # .cuda()
                self.c4_l = torch.cat(self.c4_l, dim=0)  # .cuda()
                self.c5_l = torch.cat(self.c5_l, dim=0)  # .cuda()
                self.ps00_l = torch.cat(self.ps00_l, dim=0)  # .cuda()
                self.ps01_l = torch.cat(self.ps01_l, dim=0)  # .cuda()
                self.ps02_l = torch.cat(self.ps02_l, dim=0)  # .cuda()
                self.ps03_l = torch.cat(self.ps03_l, dim=0)  # .cuda()
                self.ps10_l = torch.cat(self.ps10_l, dim=0)  # .cuda()
                self.ps11_l = torch.cat(self.ps11_l, dim=0)  # .cuda()
                self.ps12_l = torch.cat(self.ps12_l, dim=0)  # .cuda()
                self.ps13_l = torch.cat(self.ps13_l, dim=0)  # .cuda()
                self.ps20_l = torch.cat(self.ps20_l, dim=0)  # .cuda()
                self.ps21_l = torch.cat(self.ps21_l, dim=0)  # .cuda()
                self.ps22_l = torch.cat(self.ps22_l, dim=0)  # .cuda()
                self.ps23_l = torch.cat(self.ps23_l, dim=0)  # .cuda()
                self.ps0_l = [self.ps00_l, self.ps01_l, self.ps02_l, self.ps03_l]
                self.ps1_l = [self.ps10_l, self.ps11_l, self.ps12_l, self.ps13_l]
                self.ps2_l = [self.ps20_l, self.ps21_l, self.ps22_l, self.ps23_l]
                # self.ps3_l = torch.cat(self.ps3_l, dim=0)  # .cuda()
            return self.ps3_l, output  # self.output_l

    def forward(
        self,
        image_global,
        patches,
        top_lefts,
        ratio,
        mode=PhaseMode.GlobalOnly,
        n_patch=None,
    ):
        if mode is PhaseMode.GlobalOnly:
            outputs = self.backbone_global.forward(image_global)
            output_g = self.fpn_global.forward(backbone_outputs=outputs)[0]
            return output_g, None
        if mode is PhaseMode.LocalFromGlobal:
            with torch.no_grad():
                if self.patch_n == 0:
                    self.c2_g, self.c3_g, self.c4_g, self.c5_g = self.backbone_global.forward(image_global)
                    self.output_g, self.ps0_g, self.ps1_g, self.ps2_g, self.ps3_g = \
                        self.fpn_global.forward(backbone_outputs=(self.c2_g, self.c3_g, self.c4_g, self.c5_g))
                    
                self.patch_n += patches.size()[0]
                self.patch_n %= n_patch
            
            # local model
            outputs = self.backbone_local.forward(patches)
            output_l, _, _, _, ps3_l = self.fpn_local.forward(
                backbone_outputs=outputs,
                additional_fms=(
                    self._crop_global(self.c2_g, top_lefts, ratio),
                    self._crop_global(self.c3_g, top_lefts, ratio),
                    self._crop_global(self.c4_g, top_lefts, ratio),
                    self._crop_global(self.c5_g, top_lefts, ratio)
                ),
                ps_exts=(
                    [self._crop_global(f, top_lefts, ratio) for f in self.ps0_g],
                    [self._crop_global(f, top_lefts, ratio) for f in self.ps1_g],
                    [self._crop_global(f, top_lefts, ratio) for f in self.ps2_g]
                )
            )

            ps3_g2l = crop_global(self.ps3_g, top_lefts, ratio)[0]
            ps3_g2l = F.interpolate(ps3_g2l, size=ps3_l.size()[2:], **self._up_kwargs)

            output = self.ensemble(ps3_l, ps3_g2l)
            # output = F.interpolate(output, imsize, mode='nearest')
            return output, self.output_g, output_l, nn.MSELoss(ps3_l, ps3_g2l)
        if mode is PhaseMode.GlobalFromLocal:
            outputs = self.backbone_global.forward(image_global)

            output_g, _, _, _, ps3_g = self.fpn_global.forward(
                backbone_outputs=outputs,
                additional_fms=(self.c2_l, self.c3_l, self.c4_l, self.c5_l),
                ps_exts=(self.ps0_l, self.ps1_l, self.ps2_l)
            )
            self.clear_cache()
            return output_g, ps3_g
