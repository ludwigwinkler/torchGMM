import sys

import pytest
import torch
import einops


from torchGMM.diffusion import (
    forward_diffusion,
    reverse_diffusion,
)
from torchGMM.gmm import TimeDependentGMM
from torchGMM.schedule import BetaSchedule


class TestDiffusion:

    def test_forward_diffusion(self):
        pass

    def test_reverse_diffusion(self):
        pass
