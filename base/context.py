"""A starting implementation inspired by Wang and Tang (2024), Figure 6."""

import torch
from torch import nn


class ContextBlock(nn.Module):
    """Fuse neighboring context at three dilation rates, preserving input shape."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=3, padding=rate, dilation=rate)
            for rate in (1, 3, 5)
        )
        fused_channels = 3 * channels
        hidden_channels = max(fused_channels // reduction, 1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fused_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, fused_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.project = nn.Conv2d(fused_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.project(context * self.channel_gate(context))
