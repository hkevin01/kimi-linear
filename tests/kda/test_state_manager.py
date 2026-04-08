"""Tests for StateManager module."""

import pytest
import torch
from src.kda.state_manager import StateManager


@pytest.fixture
def manager():
    return StateManager(key_dim=64, value_dim=64, num_heads=8, max_batch_size=16)


def _inputs(B=4, H=8, K=64, V=64):
    keys = torch.nn.functional.normalize(torch.randn(B, H, K), dim=-1)
    values = torch.randn(B, H, V)
    gates = torch.sigmoid(torch.randn(B, H, K))
    beta = torch.sigmoid(torch.randn(B, H, 1)) * 0.5
    return keys, values, gates, beta


def test_init_shape(manager):
    state = manager.initialize_state(4)
    assert state.shape == (4, 8, 64, 64)


def test_init_zeros(manager):
    state = manager.initialize_state(4)
    assert state.abs().sum().item() == 0.0


def test_update_shape(manager):
    state = manager.initialize_state(4)
    keys, values, gates, beta = _inputs()
    new_state, _ = manager.update_state(state, keys, values, gates, beta)
    assert new_state.shape == state.shape


def test_update_timing(manager):
    state = manager.initialize_state(4)
    keys, values, gates, beta = _inputs()
    _, timing = manager.update_state(state, keys, values, gates, beta, return_timing=True)
    assert isinstance(timing, float) and timing > 0.0


def test_no_nan_after_update(manager):
    state = manager.initialize_state(4)
    keys, values, gates, beta = _inputs()
    new_state, _ = manager.update_state(state, keys, values, gates, beta)
    assert not torch.isnan(new_state).any()


def test_batch_too_large_raises(manager):
    with pytest.raises(ValueError):
        manager.initialize_state(32)  # max_batch_size=16


def test_shape_mismatch_raises(manager):
    state = manager.initialize_state(4)
    bad_keys = torch.randn(4, 8, 32)  # wrong key_dim
    values = torch.randn(4, 8, 64)
    gates = torch.sigmoid(torch.randn(4, 8, 64))
    beta = torch.sigmoid(torch.randn(4, 8, 1))
    with pytest.raises(ValueError):
        manager.update_state(state, bad_keys, values, gates, beta)


def test_checkpointing(manager):
    manager.enable_checkpointing = True
    manager.checkpoint_interval = 1
    state = manager.initialize_state(4)
    keys, values, gates, beta = _inputs()
    for step in range(3):
        state, _ = manager.update_state(state, keys, values, gates, beta, step=step)
    assert len(manager.checkpoints) > 0


def test_memory_usage(manager):
    info = manager.get_memory_usage()
    assert "buffer_mb" in info
    assert info["buffer_mb"] > 0.0
