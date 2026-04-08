"""
Integration tests for the assembled KDA stack.

Tests cover end-to-end behaviour across multiple layers, stateful chunked
processing (sequential-chunk equivalence), gradient flow through the full
pipeline, output norm stability, and state persistence via save/load.
"""

import pytest
import torch
import torch.nn as nn
import io
from src.kda.kda_layer import KDALayer


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg():
    """Shared small configuration for fast integration tests."""
    return dict(hidden_dim=64, num_heads=2, head_dim=8, max_batch_size=4)


@pytest.fixture(scope="module")
def single_layer(cfg):
    torch.manual_seed(42)
    return KDALayer(**cfg).eval()


@pytest.fixture(scope="module")
def three_layer_stack(cfg):
    torch.manual_seed(0)

    class KDAStack(nn.Module):
        def __init__(self, cfg, n=3):
            super().__init__()
            self.layers = nn.ModuleList([KDALayer(**cfg) for _ in range(n)])
            self.n = n

        def forward(self, x):
            states = []
            for layer in self.layers:
                x, state = layer(x)
                states.append(state)
            return x, states

    return KDAStack(cfg).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Test: multi-layer stack output shape
# ─────────────────────────────────────────────────────────────────────────────

def test_stack_output_shape(three_layer_stack, cfg):
    """Three stacked KDA layers produce output matching input shape."""
    B, T = 2, 16
    x = torch.randn(B, T, cfg["hidden_dim"])
    out, states = three_layer_stack(x)
    assert out.shape == (B, T, cfg["hidden_dim"]), \
        f"Expected {(B, T, cfg['hidden_dim'])}, got {out.shape}"
    assert len(states) == 3


def test_stack_no_nan(three_layer_stack, cfg):
    """No NaN in any layer output for the three-layer stack."""
    x = torch.randn(2, 8, cfg["hidden_dim"])
    out, states = three_layer_stack(x)
    assert not torch.isnan(out).any(), "NaN in stack output"
    for i, s in enumerate(states):
        assert not torch.isnan(s).any(), f"NaN in layer {i} state"


# ─────────────────────────────────────────────────────────────────────────────
# Test: chunked processing equivalence
# A full sequence processed in one pass must equal two sequential half-chunks
# when the state is carried across chunk boundaries.
# ─────────────────────────────────────────────────────────────────────────────

def test_chunked_sequence_equivalence(cfg):
    """
    Sequential chunks must produce the same output as a single full pass
    when short convolution is disabled (short conv requires cross-chunk carry).
    """
    torch.manual_seed(7)
    # Disable short conv: causal conv needs previous-chunk K values for exact equality
    layer = KDALayer(**cfg, use_short_conv=False).eval()
    B, T = 1, 10
    x = torch.randn(B, T, cfg["hidden_dim"])

    with torch.no_grad():
        # Full pass
        full_out, _ = layer(x)

        # Two half-chunks with carried state
        x1, x2 = x[:, :T // 2, :], x[:, T // 2:, :]
        out1, state1 = layer(x1)
        out2, _ = layer(x2, state=state1)
        chunked_out = torch.cat([out1, out2], dim=1)

    assert full_out.shape == chunked_out.shape
    assert torch.allclose(full_out, chunked_out, atol=1e-5), \
        f"Max diff: {(full_out - chunked_out).abs().max().item():.2e}"


def test_chunked_state_shape(single_layer, cfg):
    """State returned from chunk 1 has the correct shape for chunk 2 input."""
    x = torch.randn(2, 4, cfg["hidden_dim"])
    _, state = single_layer(x)
    H, K, V = cfg["num_heads"], cfg["head_dim"], cfg["head_dim"]
    assert state.shape == (2, H, K, V), f"Unexpected state shape: {state.shape}"


# ─────────────────────────────────────────────────────────────────────────────
# Test: gradient flow through full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def test_gradient_flow_through_stack(three_layer_stack, cfg):
    """Gradients must flow back to the input through all three KDA layers."""
    x = torch.randn(2, 6, cfg["hidden_dim"], requires_grad=True)
    out, _ = three_layer_stack(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient on input"
    assert not torch.isnan(x.grad).any(), "NaN in input gradient"
    assert x.grad.abs().sum() > 0, "Zero gradient — computation graph not connected"


def test_gradient_flow_single(single_layer, cfg):
    """Gradients must propagate to all learnable parameters."""
    layer = KDALayer(**cfg)
    x = torch.randn(1, 4, cfg["hidden_dim"])
    out, _ = layer(x)
    out.sum().backward()
    no_grad = [n for n, p in layer.named_parameters() if p.grad is None]
    assert not no_grad, f"Parameters with no gradient: {no_grad}"


# ─────────────────────────────────────────────────────────────────────────────
# Test: output RMSNorm keeps values bounded
# ─────────────────────────────────────────────────────────────────────────────

def test_output_magnitude_bounded(single_layer, cfg):
    """Output values must be bounded — RMSNorm + output gate keep them finite."""
    torch.manual_seed(1)
    x = torch.randn(2, 32, cfg["hidden_dim"]) * 10  # large input scale
    with torch.no_grad():
        out, _ = single_layer(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    # With RMSNorm the typical magnitude should be reasonably bounded
    assert out.abs().max().item() < 1e4, \
        f"Output too large: {out.abs().max().item()}"


# ─────────────────────────────────────────────────────────────────────────────
# Test: stateful accumulation — state grows monotonically in Frobenius norm
# under pure zero-gate (full forgetting overrides all) → state resets
# ─────────────────────────────────────────────────────────────────────────────

def test_state_accumulates_over_steps(single_layer, cfg):
    """State should change at each step when non-trivial input is fed."""
    torch.manual_seed(99)
    state = None
    prev_norm = 0.0
    changed = False
    for _ in range(5):
        x = torch.randn(1, 1, cfg["hidden_dim"])
        with torch.no_grad():
            _, state = single_layer(x, state=state)
        norm = state.norm().item()
        if abs(norm - prev_norm) > 1e-6:
            changed = True
        prev_norm = norm
    assert changed, "State did not change across sequential token steps"


# ─────────────────────────────────────────────────────────────────────────────
# Test: state round-trip via torch.save / torch.load
# ─────────────────────────────────────────────────────────────────────────────

def test_state_save_load_roundtrip(single_layer, cfg):
    """State saved to a buffer and loaded back must produce identical output."""
    x_init = torch.randn(1, 4, cfg["hidden_dim"])
    x_cont = torch.randn(1, 4, cfg["hidden_dim"])

    with torch.no_grad():
        _, state1 = single_layer(x_init)

        buf = io.BytesIO()
        torch.save(state1, buf)
        buf.seek(0)
        state1_loaded = torch.load(buf, weights_only=True)

        out_orig, _ = single_layer(x_cont, state=state1)
        out_loaded, _ = single_layer(x_cont, state=state1_loaded)

    assert torch.allclose(out_orig, out_loaded, atol=1e-6), \
        f"Save/load round-trip mismatch; max diff={( out_orig - out_loaded).abs().max():.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# Test: determinism — same input gives same output with fixed seed
# ─────────────────────────────────────────────────────────────────────────────

def test_deterministic_output(cfg):
    """Equal seeds and inputs must produce equal outputs."""
    def run(seed):
        torch.manual_seed(seed)
        layer = KDALayer(**cfg).eval()
        x = torch.randn(2, 8, cfg["hidden_dim"])
        torch.manual_seed(seed)  # reset for identical dropout
        with torch.no_grad():
            out, _ = layer(x)
        return out

    out1 = run(42)
    out2 = run(42)
    assert torch.allclose(out1, out2, atol=0.0), "Non-deterministic output detected"


# ─────────────────────────────────────────────────────────────────────────────
# Test: short conv disabled variant still works
# ─────────────────────────────────────────────────────────────────────────────

def test_no_short_conv(cfg):
    """KDALayer without short convolution should still produce valid output."""
    layer = KDALayer(**cfg, use_short_conv=False).eval()
    x = torch.randn(1, 8, cfg["hidden_dim"])
    with torch.no_grad():
        out, state = layer(x)
    assert out.shape == (1, 8, cfg["hidden_dim"])
    assert not torch.isnan(out).any()


def test_no_output_gate(cfg):
    """KDALayer without output gate should still produce valid output."""
    layer = KDALayer(**cfg, use_output_gate=False).eval()
    x = torch.randn(1, 8, cfg["hidden_dim"])
    with torch.no_grad():
        out, state = layer(x)
    assert out.shape == (1, 8, cfg["hidden_dim"])
    assert not torch.isnan(out).any()
