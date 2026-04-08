"""Tests for DPLRTransition module."""

import pytest
import torch
import torch.nn.functional as F
from src.kda.dplr import DPLRTransition


@pytest.fixture
def dplr():
    return DPLRTransition(key_dim=64, value_dim=64, num_heads=8)


def _inputs(B=4, H=8, K=64, V=64):
    state = torch.randn(B, H, K, V) * 0.1
    keys = F.normalize(torch.randn(B, H, K), dim=-1)
    values = torch.randn(B, H, V)
    gates = torch.sigmoid(torch.randn(B, H, K))
    beta = torch.sigmoid(torch.randn(B, H, 1)) * 0.5
    return state, keys, values, gates, beta


def test_transition_shape(dplr):
    state, keys, _, gates, beta = _inputs()
    new_state, _ = dplr.compute_transition(state, keys, gates, beta)
    assert new_state.shape == state.shape


def test_full_forward_shape(dplr):
    state, keys, values, gates, beta = _inputs()
    new_state, _ = dplr(state, keys, values, gates, beta)
    assert new_state.shape == state.shape


def test_no_nan(dplr):
    state, keys, values, gates, beta = _inputs()
    new_state, _ = dplr(state, keys, values, gates, beta)
    assert not torch.isnan(new_state).any()


def test_timing_returned(dplr):
    state, keys, values, gates, beta = _inputs()
    _, timing = dplr(state, keys, values, gates, beta, return_timing=True)
    assert isinstance(timing, float) and timing > 0.0


def test_heavy_forgetting(dplr):
    state, keys, values, gates, beta = _inputs()
    gates_zero = torch.ones_like(gates) * 0.001
    new_state, _ = dplr.compute_transition(state, keys, gates_zero, beta)
    assert torch.norm(new_state).item() < torch.norm(state).item()


def test_minimal_forgetting(dplr):
    state, keys, values, gates, beta = _inputs()
    gates_one = torch.ones_like(gates) * 0.999
    new_state, _ = dplr.compute_transition(state, keys, gates_one, beta)
    # State should not grow unboundedly
    assert not torch.isinf(new_state).any()


def test_wrong_key_shape_raises(dplr):
    state, _, values, gates, beta = _inputs()
    bad_keys = torch.randn(4, 8, 32)  # wrong key_dim
    with pytest.raises(ValueError):
        dplr.compute_transition(state, bad_keys, gates, beta)


def test_wrong_gate_shape_raises(dplr):
    state, keys, values, _, beta = _inputs()
    bad_gates = torch.randn(4, 8, 32)  # wrong dim
    with pytest.raises(ValueError):
        dplr.compute_transition(state, keys, bad_gates, beta)


def test_invalid_construction():
    with pytest.raises(ValueError):
        DPLRTransition(key_dim=0, value_dim=64, num_heads=8)
    with pytest.raises(ValueError):
        DPLRTransition(key_dim=64, value_dim=64, num_heads=0)
