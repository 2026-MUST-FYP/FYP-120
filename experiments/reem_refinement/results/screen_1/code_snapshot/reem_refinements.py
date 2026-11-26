"""P3 REEM refinements that start from the same unmodified REEM response."""

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn import tasks
from ultralytics.utils.torch_utils import initialize_weights

from base.REEM import REEM
from src.reem_p3 import DirectionalREEM, build_detector


class OpposedStripeGate(nn.Conv2d):
    """Produce complementary horizontal and vertical logits from a shared score."""

    def __init__(self):
        super().__init__(2, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        score = super().forward(x).tanh()
        return torch.cat((score, -score), dim=1)


class BalancedDirectionalREEM(DirectionalREEM):
    """Reallocate stripe response without lowering both directions together."""

    def __init__(self, channels):
        super().__init__(channels)
        self.direction_gate = OpposedStripeGate()


def ridge_profiles(x):
    """Measure thin ridges via same-polarity contrast against both sides."""
    horizontal = F.avg_pool2d(x, (1, 5), stride=1, padding=(0, 2), count_include_pad=False)
    vertical = F.avg_pool2d(x, (5, 1), stride=1, padding=(2, 0), count_include_pad=False)
    padded_h = F.pad(horizontal, (0, 0, 2, 2), mode="replicate")
    padded_v = F.pad(vertical, (2, 2, 0, 0), mode="replicate")
    responses = []
    for center, left, right in ((horizontal, padded_h[..., :-4, :], padded_h[..., 4:, :]),
                                (vertical, padded_v[..., :-4], padded_v[..., 4:])):
        delta_left, delta_right = center - left, center - right
        responses.append(torch.minimum(delta_left, delta_right).relu() +
                         torch.minimum(-delta_left, -delta_right).relu())
    return torch.cat(responses, dim=1)


class RidgeContrastREEM(REEM):
    """Add a signed residual from horizontal and vertical two-sided contrast."""

    def __init__(self, channels):
        super().__init__(channels)
        self.ridge_norm = nn.BatchNorm2d(2 * channels)
        self.ridge_project = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.ridge_project.weight)

    def forward(self, x):
        contrast = self.ridge_project(self.ridge_norm(ridge_profiles(x)))
        return super().forward(x) + contrast


REFINEMENTS = {"balanced_reem": BalancedDirectionalREEM, "ridge_reem": RidgeContrastREEM}


def build_refinement(source, variant, seed, verbose=False):
    """Change only the inserted P3 block while copying the same REEM state."""
    detector = build_detector(source, "original_reem", seed, verbose=False)
    if variant == "original_reem":
        return detector

    original = detector.model[-2]
    cls = REFINEMENTS[variant]
    setattr(tasks, cls.__name__, cls)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        refinement = cls(original.channels)
    initialize_weights(refinement)
    own_state = refinement.state_dict()
    original_state = original.state_dict()
    own_state.update(original_state)
    refinement.load_state_dict(own_state, strict=True)
    refinement.i, refinement.f = original.i, original.f
    refinement.type = f"{cls.__module__}.{cls.__name__}"
    refinement.np = sum(p.numel() for p in refinement.parameters())
    detector.model[-2] = refinement
    detector.yaml["head"][-2][2] = cls.__name__
    if verbose:
        detector.info()
    return detector
