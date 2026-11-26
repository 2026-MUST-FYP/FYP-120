"""Add short-range ridge evidence while accounting for long-line continuity."""

import torch
from torch.nn import functional as F

from base.REEM import REEM
from src.reem_refinements import BalancedDirectionalREEM, RidgeContrastREEM, ridge_profiles


def localized_ridges(x):
    horizontal, vertical = ridge_profiles(x).chunk(2, dim=1)
    persistent_h = F.avg_pool2d(horizontal, (1, 9), stride=1, padding=(0, 4), count_include_pad=False)
    persistent_v = F.avg_pool2d(vertical, (9, 1), stride=1, padding=(4, 0), count_include_pad=False)
    return horizontal - persistent_h, vertical - persistent_v


class LocalRidgeREEM(RidgeContrastREEM):
    """Feed short- and long-line contrast to the residual branch as a difference."""

    def forward(self, x):
        contrast = torch.cat(localized_ridges(x), dim=1)
        return REEM.forward(self, x) + self.ridge_project(self.ridge_norm(contrast))


class LineConditionedREEM(BalancedDirectionalREEM):
    """Use local ridge evidence to choose the weights of the existing stripes."""

    def forward(self, x):
        horizontal, vertical = self.strip_h(x), self.strip_v(x)
        ridge_h, ridge_v = localized_ridges(x)
        direction_input = torch.cat((ridge_h.mean(dim=1, keepdim=True), ridge_v.mean(dim=1, keepdim=True)), dim=1)
        gain_h, gain_v = (2 * self.direction_gate(direction_input).sigmoid()).split(1, dim=1)
        sobel = self.edge_norm(self._sobel_magnitude(x))
        features = torch.cat((self.local_dw(x), self.dilate_dw(x), horizontal * gain_h, vertical * gain_v, sobel), dim=1)
        edge = self.fuse(features)
        edge = edge * self.channel_gate(edge) * self.spatial_gate(edge)
        return x + torch.tanh(self.alpha) * edge


LINE_REFINEMENTS = {"local_ridge_reem": LocalRidgeREEM, "line_conditioned_reem": LineConditionedREEM}
