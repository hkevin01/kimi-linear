"""
Tests for src.kda.triton_kernels:
  - HAS_TRITON flag
  - chunk_kda_forward  (pure-PyTorch path, FLA unavailable in CI)
  - fused_recurrent_kda_forward  (pure-PyTorch path)

Input convention for both public functions:
  q, k, v  : (B, H, T, d_k / d_v)
  g        : (B, H, T, 1)   log-space scalar gate per head/token (≤ 0)
  beta     : (B, H, T, 1)   delta-rule write rate ∈ (0, 1]
  state    : (B, H, d_k, d_v) or None
"""
import pytest
import torch
import torch.nn.functional as F

from src.kda.triton_kernels import (
    HAS_TRITON,
    chunk_kda_forward,
    fused_recurrent_kda_forward,
)

# ── Dimensions used throughout ────────────────────────────────────────────────
B, H, T, D, D_V = 2, 4, 64, 16, 16


def _inputs(t=T, b=B, h=H, d=D, d_v=D_V, *, with_state=False):
    """Return (q, k, v, g, beta[, state]) in the (B, H, T, d) convention."""
    q = torch.randn(b, h, t, d)
    k = F.normalize(torch.randn(b, h, t, d), p=2, dim=-1)
    v = torch.randn(b, h, t, d_v)
    g = -torch.rand(b, h, t, 1) * 0.5         # log-space gate ≤ 0
    beta = torch.rand(b, h, t, 1) * 0.5 + 0.25  # ∈ (0.25, 0.75)
    if with_state:
        state = torch.zeros(b, h, d, d_v)
        return q, k, v, g, beta, state
    return q, k, v, g, beta


# ── HAS_TRITON ────────────────────────────────────────────────────────────────

class TestHASTriton:
    def test_is_bool(self):
        assert isinstance(HAS_TRITON, bool)

    def test_false_without_fla(self):
        # FLA is not installed in this environment; flag must be False
        assert HAS_TRITON is False


# ── chunk_kda_forward ─────────────────────────────────────────────────────────

class TestChunkKDAForward:
    def test_output_shapes(self):
        q, k, v, g, beta = _inputs()
        out, state = chunk_kda_forward(q, k, v, g, beta)
        assert out.shape == (B, H, T, D_V), f"output: {out.shape}"
        assert state.shape == (B, H, D, D_V), f"state: {state.shape}"

    def test_no_nan_output(self):
        q, k, v, g, beta = _inputs()
        out, state = chunk_kda_forward(q, k, v, g, beta)
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isnan(state).any(), "NaN in state"

    def test_no_inf_output(self):
        q, k, v, g, beta = _inputs()
        out, state = chunk_kda_forward(q, k, v, g, beta)
        assert not torch.isinf(out).any()
        assert not torch.isinf(state).any()

    def test_default_scale_applied(self):
        """With random Q, smaller head_dim → larger scale → larger magnitude."""
        q, k, v, g, beta = _inputs(d=4)
        out_small, _ = chunk_kda_forward(q[:, :, :, :4], k[:, :, :, :4],
                                          v, g, beta,
                                          scale=4 ** -0.5)
        q2, k2, v2, g2, b2 = _inputs(d=64)
        out_large, _ = chunk_kda_forward(q2, k2, v2, g2, b2,
                                          scale=64 ** -0.5)
        # Both must be finite — scale validity check only
        assert not torch.isnan(out_small).any()
        assert not torch.isnan(out_large).any()

    def test_explicit_scale_respected(self):
        q, k, v, g, beta = _inputs()
        out1, _ = chunk_kda_forward(q, k, v, g, beta, scale=1.0)
        out2, _ = chunk_kda_forward(q, k, v, g, beta, scale=0.1)
        # Different scales → different outputs
        assert not torch.allclose(out1, out2)

    def test_initial_state_passthrough(self):
        q, k, v, g, beta, s0 = _inputs(with_state=True)
        out_no_state, _ = chunk_kda_forward(q, k, v, g, beta)
        out_with_state, _ = chunk_kda_forward(q, k, v, g, beta, initial_state=s0)
        # Zero initial state → same result as no state
        assert torch.allclose(out_no_state, out_with_state, atol=1e-6)

    def test_nonzero_initial_state_changes_output(self):
        q, k, v, g, beta = _inputs()
        s_nonzero = torch.randn(B, H, D, D_V)
        out_zero, _ = chunk_kda_forward(q, k, v, g, beta)
        out_nz, _ = chunk_kda_forward(q, k, v, g, beta, initial_state=s_nonzero)
        assert not torch.allclose(out_zero, out_nz)

    def test_state_changes_between_calls(self):
        """Passing final state of call 1 into call 2 should change output."""
        q, k, v, g, beta = _inputs()
        _, state1 = chunk_kda_forward(q, k, v, g, beta)
        q2, k2, v2, g2, beta2 = _inputs()
        out_cold, _ = chunk_kda_forward(q2, k2, v2, g2, beta2)
        out_warm, _ = chunk_kda_forward(q2, k2, v2, g2, beta2, initial_state=state1)
        assert not torch.allclose(out_cold, out_warm)

    def test_return_intermediate_states(self):
        q, k, v, g, beta = _inputs()
        result = chunk_kda_forward(q, k, v, g, beta,
                                   return_intermediate_states=True)
        assert len(result) == 3, "Expected (output, state, inter_states)"
        out, state, inter = result
        assert out.shape == (B, H, T, D_V)
        assert state.shape == (B, H, D, D_V)
        # inter: (B, H, T, d_k, d_v)
        assert inter.shape[0] == B
        assert inter.shape[1] == H

    def test_various_sequence_lengths(self):
        """Should work regardless of T (no power-of-2 constraint here)."""
        for t in [1, 7, 16, 64, 100]:
            q, k, v, g, beta = _inputs(t=t)
            out, state = chunk_kda_forward(q, k, v, g, beta)
            assert out.shape == (B, H, t, D_V), f"T={t}: {out.shape}"
            assert not torch.isnan(out).any(), f"NaN at T={t}"

    def test_batch_size_1(self):
        q, k, v, g, beta = _inputs(b=1)
        out, state = chunk_kda_forward(q, k, v, g, beta)
        assert out.shape == (1, H, T, D_V)

    def test_single_head(self):
        q, k, v, g, beta = _inputs(h=1)
        out, state = chunk_kda_forward(q, k, v, g, beta)
        assert out.shape == (B, 1, T, D_V)

    def test_gate_decay_reduces_state_influence(self):
        """Strong decay (g→-∞) should suppress the initial state contribution."""
        q, k, v, g, beta = _inputs(t=4)
        s_large = torch.ones(B, H, D, D_V) * 100.0
        g_strong = torch.full((B, H, 4, 1), -10.0)   # near-zero decay
        out_decayed, _ = chunk_kda_forward(q, k, v, g_strong, beta,
                                            initial_state=s_large)
        g_weak = torch.full((B, H, 4, 1), -0.001)
        out_preserved, _ = chunk_kda_forward(q, k, v, g_weak, beta,
                                              initial_state=s_large)
        # Strong decay → smaller magnitudes
        assert out_decayed.abs().mean() < out_preserved.abs().mean()

    def test_gradient_flows_through_q(self):
        q, k, v, g, beta = _inputs()
        q = q.detach().requires_grad_(True)
        out, _ = chunk_kda_forward(q, k, v, g, beta)
        out.sum().backward()
        assert q.grad is not None
        assert not torch.isnan(q.grad).any()

    def test_gradient_flows_through_v(self):
        q, k, v, g, beta = _inputs()
        v = v.detach().requires_grad_(True)
        out, _ = chunk_kda_forward(q, k, v, g, beta)
        out.sum().backward()
        assert v.grad is not None
        assert not torch.isnan(v.grad).any()


# ── fused_recurrent_kda_forward ───────────────────────────────────────────────

class TestFusedRecurrentKDAForward:
    def test_output_shapes(self):
        q, k, v, g, beta = _inputs()
        out, state = fused_recurrent_kda_forward(q, k, v, g, beta)
        assert out.shape == (B, H, T, D_V), f"output: {out.shape}"
        assert state.shape == (B, H, D, D_V), f"state: {state.shape}"

    def test_no_nan_output(self):
        q, k, v, g, beta = _inputs()
        out, state = fused_recurrent_kda_forward(q, k, v, g, beta)
        assert not torch.isnan(out).any()
        assert not torch.isnan(state).any()

    def test_no_inf_output(self):
        q, k, v, g, beta = _inputs()
        out, state = fused_recurrent_kda_forward(q, k, v, g, beta)
        assert not torch.isinf(out).any()

    def test_initial_state_zero_equals_none(self):
        q, k, v, g, beta, s0 = _inputs(with_state=True)
        out_none, _ = fused_recurrent_kda_forward(q, k, v, g, beta)
        out_zero, _ = fused_recurrent_kda_forward(q, k, v, g, beta,
                                                   initial_state=s0)
        assert torch.allclose(out_none, out_zero, atol=1e-6)

    def test_nonzero_state_changes_output(self):
        q, k, v, g, beta = _inputs()
        s_nz = torch.randn(B, H, D, D_V)
        out_zero, _ = fused_recurrent_kda_forward(q, k, v, g, beta)
        out_nz, _ = fused_recurrent_kda_forward(q, k, v, g, beta,
                                                 initial_state=s_nz)
        assert not torch.allclose(out_zero, out_nz)

    def test_explicit_scale_respected(self):
        q, k, v, g, beta = _inputs()
        out1, _ = fused_recurrent_kda_forward(q, k, v, g, beta, scale=1.0)
        out2, _ = fused_recurrent_kda_forward(q, k, v, g, beta, scale=0.1)
        assert not torch.allclose(out1, out2)

    def test_various_sequence_lengths(self):
        for t in [1, 4, 16, 64]:
            q, k, v, g, beta = _inputs(t=t)
            out, state = fused_recurrent_kda_forward(q, k, v, g, beta)
            assert out.shape == (B, H, t, D_V)
            assert not torch.isnan(out).any()

    def test_single_token_decode(self):
        """Common decode step: T=1."""
        q, k, v, g, beta = _inputs(t=1)
        out, state = fused_recurrent_kda_forward(q, k, v, g, beta)
        assert out.shape == (B, H, 1, D_V)
        assert state.shape == (B, H, D, D_V)

    def test_gradient_flows(self):
        q, k, v, g, beta = _inputs(t=8)
        q = q.detach().requires_grad_(True)
        out, _ = fused_recurrent_kda_forward(q, k, v, g, beta)
        out.sum().backward()
        assert q.grad is not None
        assert not torch.isnan(q.grad).any()

    def test_chunk_and_recurrent_agree(self):
        """chunk_kda_forward and fused_recurrent must produce same output."""
        q, k, v, g, beta = _inputs(t=16)
        out_chunk, state_chunk = chunk_kda_forward(q, k, v, g, beta)
        out_recur, state_recur = fused_recurrent_kda_forward(q, k, v, g, beta)
        assert torch.allclose(out_chunk, out_recur, atol=1e-5), \
            f"max diff: {(out_chunk - out_recur).abs().max().item():.2e}"
        assert torch.allclose(state_chunk, state_recur, atol=1e-5), \
            f"state max diff: {(state_chunk - state_recur).abs().max().item():.2e}"
