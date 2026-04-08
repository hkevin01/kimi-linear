"""Tests for FineGrainedGating module."""

import pytest
import torch
from src.kda.gating import FineGrainedGating


@pytest.fixture
def gating():
    return FineGrainedGating(hidden_dim=512, head_dim=64, num_heads=8)


def test_output_shape(gating):
    x = torch.randn(4, 32, 512)
    gates, _ = gating(x)
    assert gates.shape == (4, 32, 8, 64)


def test_gates_in_unit_range(gating):
    x = torch.randn(4, 32, 512)
    gates, _ = gating(x)
    assert gates.min().item() >= 0.0
    assert gates.max().item() <= 1.0


def test_timing_returned(gating):
    x = torch.randn(2, 16, 512)
    gates, timing = gating(x, return_timing=True)
    assert isinstance(timing, float)
    assert timing > 0.0


def test_wrong_input_dim_raises(gating):
    x = torch.randn(4, 512)  # missing seq_len dim
    with pytest.raises(ValueError):
        gating(x)


def test_wrong_hidden_dim_raises(gating):
    x = torch.randn(4, 32, 256)  # wrong hidden dim
    with pytest.raises(ValueError):
        gating(x)


def test_small_input(gating):
    x = torch.randn(1, 1, 512) * 1e-6
    gates, _ = gating(x)
    assert gates.shape == (1, 1, 8, 64)
    assert not torch.isnan(gates).any()


def test_large_input(gating):
    x = torch.randn(1, 1, 512) * 1e3
    gates, _ = gating(x)
    assert not torch.isnan(gates).any()
    assert gates.min().item() >= 0.0
    assert gates.max().item() <= 1.0


def test_performance_tracking(gating):
    x = torch.randn(2, 16, 512)
    gating(x, return_timing=True)
    gating(x, return_timing=True)
    assert gating.forward_calls == 2
    assert gating.get_average_time() > 0.0


def test_invalid_construction():
    with pytest.raises(ValueError):
        FineGrainedGating(hidden_dim=0, head_dim=64, num_heads=8)
    with pytest.raises(ValueError):
        FineGrainedGating(hidden_dim=512, head_dim=-1, num_heads=8)
