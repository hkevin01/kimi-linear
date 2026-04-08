"""Tests for the assembled KDALayer module."""

import pytest
import torch
from src.kda.kda_layer import KDALayer


@pytest.fixture
def layer():
    return KDALayer(hidden_dim=128, head_dim=16, num_heads=4, max_batch_size=8)


def test_output_shape(layer):
    x = torch.randn(2, 8, 128)
    out, state = layer(x)
    assert out.shape == (2, 8, 128)
    assert state.shape == (2, 4, 16, 16)


def test_no_nan(layer):
    x = torch.randn(2, 8, 128)
    out, state = layer(x)
    assert not torch.isnan(out).any()
    assert not torch.isnan(state).any()


def test_stateful_continuation(layer):
    x1 = torch.randn(2, 4, 128)
    x2 = torch.randn(2, 4, 128)
    out1, state1 = layer(x1)
    out2_with_state, _ = layer(x2, state=state1)
    out2_no_state, _ = layer(x2)
    # Outputs should differ when initial state differs
    assert not torch.allclose(out2_with_state, out2_no_state)


def test_single_token(layer):
    x = torch.randn(1, 1, 128)
    out, state = layer(x)
    assert out.shape == (1, 1, 128)


def test_invalid_dims():
    with pytest.raises(ValueError):
        KDALayer(hidden_dim=0, head_dim=16, num_heads=4)


def test_gradient_flow(layer):
    x = torch.randn(2, 4, 128, requires_grad=True)
    out, _ = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
