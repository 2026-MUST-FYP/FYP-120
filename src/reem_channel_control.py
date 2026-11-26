"""Reallocate stripe response per channel using short- and long-line evidence."""

import torch
from torch import nn

from base.REEM import REEM
from src.reem_line_control import localized_ridges


class ChannelLineREEM(REEM):
    """Use each channel's own stripe energy and local ridge contrast."""

    def __init__(self, channels):
        super().__init__(channels)
        self.direction_gate = nn.Conv2d(4 * channels, channels, 3, padding=1, groups=channels)
        nn.init.zeros_(self.direction_gate.weight)
        nn.init.zeros_(self.direction_gate.bias)

    def forward(self, x):
        horizontal, vertical = self.strip_h(x), self.strip_v(x)
        ridge_h, ridge_v = localized_ridges(x)
        response = torch.stack((horizontal.abs(), vertical.abs(), ridge_h, ridge_v), dim=2).flatten(1, 2)
        balance = 0.5 * self.direction_gate(response).tanh()
        sobel = self.edge_norm(self._sobel_magnitude(x))
        features = torch.cat((self.local_dw(x), self.dilate_dw(x), horizontal * (1 + balance),
                              vertical * (1 - balance), sobel), dim=1)
        edge = self.fuse(features)
        edge = edge * self.channel_gate(edge) * self.spatial_gate(edge)
        return x + torch.tanh(self.alpha) * edge


CHANNEL_REFINEMENTS = {"channel_line_reem": ChannelLineREEM}
