"""Feed subpixel contrast from the higher-resolution P2 feature to P3 REEM."""

import torch
from torch import nn
from torch.nn import functional as F

from base.REEM import REEM
from src.reem_refinements import build_refinement


class DetailFusionREEM(REEM):
    """Retain P2 horizontal, vertical and diagonal differences at P3 resolution."""

    def __init__(self, channels):
        super().__init__(channels)
        self.detail_norm = nn.BatchNorm2d(3 * channels)
        self.detail_projection = nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.detail_projection.weight)

    def forward(self, inputs):
        coarse, fine = inputs
        patches = F.pixel_unshuffle(fine, 2).unflatten(1, (self.channels, 4))
        a, b, c, d = patches.unbind(2)
        detail = .5 * torch.cat((a + b - c - d, a + c - b - d, a + d - b - c), dim=1)
        residual = self.detail_projection(self.detail_norm(detail))
        return super().forward(coarse) + residual


DETAIL_VARIANT = "detail_fusion_reem"


def build_detail_detector(source, seed, verbose=False):
    detector = build_refinement(source, DETAIL_VARIANT, seed, False, {DETAIL_VARIANT: DetailFusionREEM})
    block = detector.model[-2]
    if source.model[2].cv2.conv.out_channels != block.channels:
        raise ValueError("This P2 and P3 feature pair must have identical channel counts")
    block.f = [16, 2]
    detector.save = sorted(set(detector.save + [2]))
    detector.yaml["head"][-2][0] = block.f.copy()
    if verbose:
        detector.info()
    return detector
