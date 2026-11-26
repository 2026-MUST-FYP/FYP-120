"""P3-only REEM variants for the tunnel crack detector."""

from copy import deepcopy

import torch
from torch import nn
from ultralytics.nn import tasks
from ultralytics.nn.modules import Detect

from base.REEM import REEM


class DirectionalREEM(REEM):
    """Learn one weight per location and direction before stripe fusion."""

    def __init__(self, channels):
        super().__init__(channels)
        self.direction_gate = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.direction_gate.weight)
        nn.init.zeros_(self.direction_gate.bias)

    def forward(self, x):
        local = self.local_dw(x)
        dilated = self.dilate_dw(x)
        horizontal, vertical = self.strip_h(x), self.strip_v(x)
        stripe_response = torch.cat(
            [horizontal.abs().mean(dim=1, keepdim=True), vertical.abs().mean(dim=1, keepdim=True)],
            dim=1,
        )
        horizontal_gain, vertical_gain = (2 * self.direction_gate(stripe_response).sigmoid()).split(1, dim=1)
        sobel = self.edge_norm(self._sobel_magnitude(x))
        edge = self.fuse(torch.cat([local, dilated, horizontal * horizontal_gain, vertical * vertical_gain, sobel], dim=1))
        edge = edge * self.channel_gate(edge) * self.spatial_gate(edge)
        return x + torch.tanh(self.alpha) * edge


VARIANTS = {"baseline": None, "original_reem": REEM, "directional_reem": DirectionalREEM}


def build_detector(source, variant, seed, verbose=False):
    """Keep stock P4/P5 paths and pretrained state identical across all variants."""
    block = VARIANTS[variant]
    cfg = deepcopy(source.yaml)
    if cfg.get("nc") != 1 or cfg.get("scale") != "n" or not isinstance(source.model[-1], Detect):
        raise ValueError("Expected a one-class YOLO11n source detector")
    if source.model[-1].f != [16, 19, 22] or len(source.model) != 24:
        raise ValueError("Unexpected YOLO11n feature graph")

    if block is not None:
        setattr(tasks, block.__name__, block)
        cfg["head"].insert(-1, [16, 1, block.__name__, [64]])
        cfg["head"][-1][0] = [23, 19, 22]

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        detector = tasks.DetectionModel(cfg, nc=1, verbose=verbose)

    for original, target in zip(source.model[:-1], detector.model[:23], strict=True):
        target.load_state_dict(original.state_dict(), strict=True)
    detector.model[-1].load_state_dict(source.model[-1].state_dict(), strict=True)
    return detector
