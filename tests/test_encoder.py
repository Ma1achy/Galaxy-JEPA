"""Property tests for the Encoder frozen-state helpers (core/encoder.py).

Pins the "encoder frozen during probing" hard invariant: ``assert_frozen`` catches a
trainable module and passes a frozen one.
"""

import pytest
import torch
from torch import nn

from galaxy_jepa.core.encoder import assert_frozen, is_frozen


def test_assert_frozen_catches_trainable_module():
    module = nn.Linear(4, 4)  # parameters require grad by default
    assert is_frozen(module) is False
    with pytest.raises(RuntimeError):
        assert_frozen(module)


def test_assert_frozen_passes_frozen_module():
    module = nn.Linear(4, 4)
    for param in module.parameters():
        param.requires_grad_(False)
    assert is_frozen(module) is True
    assert_frozen(module)  # must not raise


def test_module_without_parameters_is_vacuously_frozen():
    module = nn.Flatten()  # no learnable parameters
    assert is_frozen(module) is True


def test_is_frozen_requires_parameters_method():
    with pytest.raises(TypeError):
        is_frozen(object())


def test_frozen_then_forward_runs():
    module = nn.Linear(4, 4)
    for param in module.parameters():
        param.requires_grad_(False)
    assert_frozen(module)
    out = module(torch.zeros(2, 4))
    assert out.shape == (2, 4)
