import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional

from .vssm_encoder import VSSMEncoder
from .mobileone import mobileone, reparameterize_model
from .lraspp import LRASPP
from .decoder import RecurrentDecoder, Projection, ConvGRU
from .fast_guided_filter import FastGuidedFilterRefiner
from .deep_guided_filter import DeepGuidedFilterRefiner
# from .mobilevit import MobileViTEncoder


from torchvision import transforms
from pathlib import Path
from PIL import Image


class MattingNetwork(nn.Module):
    def __init__(self,
                 variant: str = 'mobilenetv3',
                 refiner: str = 'deep_guided_filter',
                 pretrained_backbone: bool = True):
        super().__init__()
        assert variant in ['mobileone', 's1', 'mobilevit', 'mamba']
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        self.variant = variant

        if variant == 'mobileone':
            self.backbone = mobileone(variant='s0')
            if pretrained_backbone:
                checkpoint = torch.load('model/weights/mobileone_s0_unfused.pth.tar')
                self.backbone.load_state_dict(checkpoint,   strict=False)
            self.aspp = LRASPP(1024, 512)
            self.decoder = RecurrentDecoder(feature_channels=[48, 48, 128, 256, 512],
                                            decoder_channels=[512, 256, 128, 16])
        elif variant == 'mamba':
            self.stem = StemHead(in_channels=3, out_channels=64)
            self.backbone = VSSMEncoder()
            self.aspp = LRASPP(768, 256)
            self.decoder = RecurrentDecoder(feature_channels=[64, 96, 192, 384, 256], decoder_channels=[256, 384, 192, 16])

        elif variant == 's1':
            self.backbone = mobileone(variant='s1')
            if pretrained_backbone:
                checkpoint = torch.load('model/weights/mobileone_s1_unfused.pth.tar')
                self.backbone.load_state_dict(checkpoint, strict=False)
            self.aspp = LRASPP(1280, 256)
            self.decoder = RecurrentDecoder(feature_channels=[64, 96, 192, 512, 256],
                                            decoder_channels=[256, 512, 256, 16])

        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)

        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()

    def forward(self,
                src: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                r5: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):

        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src

        if self.variant=='mobileone' or self.variant == 's1':
            f1, f2, f3, f4, f5 = self.backbone(src_sm)
            f5 = self.aspp(f5)
            hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, f5, r1, r2, r3, r4, r5)

        elif self.variant == 'mamba':
            f1 = self.stem(src_sm)
            f2, f3, f4, f5 = self.backbone(src_sm)
            f5 = self.aspp(f5)
            hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, f5, r1, r2, r3, r4, r5)

        if not segmentation_pass:
            fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
            if downsample_ratio != 1:
                fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            fgr = fgr_residual + src
            fgr = fgr.clamp(0., 1.)
            pha = pha.clamp(0., 1.)
            return [fgr, pha, *rec]
        else:
            seg = self.project_seg(hid)
            return [seg, *rec]

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                              mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                              mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x


class MViTHEAD(nn.Module):
    def __init__(self, features: list = None):
        super(MViTHEAD, self).__init__()
        if features is None:
            features = [32, 64, 96, 128, 640, 1024]
        self.project_f1 = Projection(in_channels=features[0], out_channels=features[1])
        self.project_f2 = Projection(in_channels=features[1], out_channels=features[2])
        self.project_f3 = Projection(in_channels=features[2], out_channels=features[3])
        self.project_f4 = Projection(in_channels=features[3], out_channels=features[4])
        self.project_f5 = Projection(in_channels=features[4], out_channels=features[5])

    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.project_f1(x1)
        x2 = self.project_f2(x2)
        x3 = self.project_f3(x3)
        x4 = self.project_f4(x4)
        x5 = self.project_f5(x5)
        return x1, x2, x3, x4, x5


class StemHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StemHead, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True),
        )

    def forward_single_frame(self, x):
        return self.stem(x)

    def forward_time_series(self, x):
        B, T = x.shape[:2]
        features = self.forward_single_frame(x.flatten(0, 1))
        features = features.unflatten(0, (B, T))
        return features

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)


if __name__ == "__main__":
    from torchsummary import summary

    _ROOT_ = Path(__file__).parents[2]
    transform = transforms.Compose([transforms.Resize((256, 512)), transforms.ToTensor()])
    img_path = "/home/sergi-garcia/Projects/Finetunning/matting-data/HD/Brainstorm/0000/com/0000.png"
    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0).unsqueeze(0).to('cuda')
    model = MattingNetwork(variant='mobileone').to('cuda')
    model = reparameterize_model(model)
    summary(model, (3, 512, 512))
    # features = model(img)
    # x = 2




