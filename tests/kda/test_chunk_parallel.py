"""
Tests for src.kda.chunk_parallel:
  - ChunkwiseParallelKDA
  - compute_wy_representation
  - ut_transform_state_update
  - compute_intra_chunk_output
"""
import pytest
import torch
import torch.nn.functional as F

from src.kda.chunk_parallel import (
    ChunkwiseParallelKDA,
    compute_wy_representation,
    compute_intra_chunk_output,
    ut_transform_state_update,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

B, H, BT, K, V = 2, 4, 8, 16, 16


def _small_inputs(chunk_size=BT):
    """Return (Q, K, V, g_log, beta) for a 2-chunk sequence."""
    T = chunk_size * 2
    Q = torch.randn(B, T, H, K)
    K_ = F.normalize(torch.randn(B, T, H, K), p=2, dim=-1)
    V_ = torch.randn(B, T, H, V)
    g_log = -torch.rand(B, T, H, K) * 0.5   # strictly negative (decay)
    beta = torch.rand(B, T, H) * 0.5 + 0.25  # ∈ (0.25, 0.75)
    return Q, K_, V_, g_log, beta


def _chunk_inputs(bt=BT):
    """Return per-chunk tensors: (B,H,BT,K), (B,H,BT,V), (B,H,BT,K), (B,H,BT)."""
    k = F.normalize(torch.randn(B, H, bt, K), p=2, dim=-1)
    v = torch.randn(B, H, bt, V)
    g = -torch.rand(B, H, bt, K) * 0.5
    beta = torch.rand(B, H, bt) * 0.5 + 0.25
    return k, v, g, beta


# ── compute_wy_representation ─────────────────────────────────────────────────

class TestComputeWYRepresentation:
    def test_output_shapes(self):
        k, v, g, beta = _chunk_inputs()
        w, y, cum_g, A = compute_wy_representation(k, v, g, beta)

        assert w.shape == (B, H, BT, K), f"w: {w.shape}"
        assert y.shape == (B, H, BT, V), f"y: {y.shape}"
        assert cum_g.shape == (B, H, K), f"cum_g: {cum_g.shape}"
        assert A.shape == (B, H, BT, BT), f"A: {A.shape}"

    def test_A_strictly_lower_triangular(self):
        k, v, g, beta = _chunk_inputs()
        _, _, _, A = compute_wy_representation(k, v, g, beta)

        # Upper triangle (including diagonal) must be zero
        upper = torch.triu(A, diagonal=0)
        assert upper.abs().max().item() < 1e-6, "A has non-zero upper triangle"

    def test_no_nan(self):
        k, v, g, beta = _chunk_inputs()
        w, y, cum_g, A = compute_wy_representation(k, v, g, beta)
        for name, t in [("w", w), ("y", y), ("cum_g", cum_g), ("A", A)]:
            assert not torch.isnan(t).any(), f"NaN in {name}"

    def test_no_inf(self):
        k, v, g, beta = _chunk_inputs()
        w, y, cum_g, A = compute_wy_representation(k, v, g, beta)
        for name, t in [("w", w), ("y", y), ("cum_g", cum_g)]:
            assert not torch.isinf(t).any(), f"Inf in {name}"

    def test_cum_g_is_cumsum(self):
        k, v, g, beta = _chunk_inputs()
        _, _, cum_g, _ = compute_wy_representation(k, v, g, beta)
        # cum_g should equal sum of g over BT dimension
        expected = g.sum(dim=2)  # (B, H, K)
        assert torch.allclose(cum_g, expected, atol=1e-5), "cum_g mismatch"


# ── ut_transform_state_update ─────────────────────────────────────────────────

class TestUTTransformStateUpdate:
    def test_output_shape(self):
        state = torch.zeros(B, H, K, V)
        w = torch.randn(B, H, BT, K)
        y = torch.randn(B, H, BT, V)
        cum_g = -torch.rand(B, H, K) * 0.5
        new_state = ut_transform_state_update(state, w, y, cum_g)
        assert new_state.shape == (B, H, K, V)

    def test_zero_state_zero_cum_g(self):
        state = torch.zeros(B, H, K, V)
        w = torch.ones(B, H, BT, K) * 0.1
        y = torch.ones(B, H, BT, V) * 0.1
        cum_g = torch.zeros(B, H, K)
        new_state = ut_transform_state_update(state, w, y, cum_g)
        # When state=0 and decay=1, new_state = sum_t w_t ⊗ y_t^T
        expected_delta = torch.einsum("bhtk,bhtv->bhkv", w, y)
        assert torch.allclose(new_state, expected_delta, atol=1e-5)

    def test_no_nan(self):
        state = torch.randn(B, H, K, V)
        k, v, g, beta = _chunk_inputs()
        w, y, cum_g, _ = compute_wy_representation(k, v, g, beta)
        new_state = ut_transform_state_update(state, w, y, cum_g)
        assert not torch.isnan(new_state).any()


# ── compute_intra_chunk_output ────────────────────────────────────────────────

class TestComputeIntraChunkOutput:
    def test_output_shape(self):
        q = torch.randn(B, H, BT, K)
        k = F.normalize(torch.randn(B, H, BT, K), p=2, dim=-1)
        g = -torch.rand(B, H, BT, K) * 0.5
        y = torch.randn(B, H, BT, V)
        o = compute_intra_chunk_output(q, k, g, y, scale=K ** -0.5)
        assert o.shape == (B, H, BT, V)

    def test_no_nan(self):
        q = torch.randn(B, H, BT, K)
        k = F.normalize(torch.randn(B, H, BT, K), p=2, dim=-1)
        g = -torch.rand(B, H, BT, K) * 0.5
        y = torch.randn(B, H, BT, V)
        o = compute_intra_chunk_output(q, k, g, y, scale=K ** -0.5)
        assert not torch.isnan(o).any()

    def test_first_token_output_zero(self):
        """Token 0 has no prior tokens, so intra-chunk output must be 0."""
        q = torch.randn(B, H, BT, K)
        k = F.normalize(torch.randn(B, H, BT, K), p=2, dim=-1)
        g = -torch.rand(B, H, BT, K) * 0.5
        y = torch.randn(B, H, BT, V)
        o = compute_intra_chunk_output(q, k, g, y, scale=K ** -0.5)
        assert o[:, :, 0, :].abs().max().item() < 1e-6, "First token intra output != 0"


# ── ChunkwiseParallelKDA ──────────────────────────────────────────────────────

class TestChunkwiseParallelKDA:
    def setup_method(self):
        self.model = ChunkwiseParallelKDA(
            key_dim=K, value_dim=V, num_heads=H, chunk_size=BT
        )

    def test_output_shapes(self):
        Q, K_, V_, g_log, beta = _small_inputs()
        o, final_state = self.model(Q, K_, V_, g_log, beta)
        T = Q.shape[1]
        assert o.shape == (B, T, H, V), f"output shape: {o.shape}"
        assert final_state.shape == (B, H, K, V), f"state shape: {final_state.shape}"

    def test_no_nan_output(self):
        Q, K_, V_, g_log, beta = _small_inputs()
        o, state = self.model(Q, K_, V_, g_log, beta)
        assert not torch.isnan(o).any(), "NaN in output"
        assert not torch.isnan(state).any(), "NaN in state"

    def test_state_passthrough(self):
        """Passing final state of step 1 as initial state of step 2 should work."""
        Q, K_, V_, g_log, beta = _small_inputs()
        _, state1 = self.model(Q, K_, V_, g_log, beta)
        Q2, K2, V2, g2, b2 = _small_inputs()
        o2, state2 = self.model(Q2, K2, V2, g2, b2, state=state1)
        assert o2.shape[0] == B
        assert not torch.isnan(o2).any()

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError):
            ChunkwiseParallelKDA(key_dim=K, value_dim=V, num_heads=H, chunk_size=3)

    def test_t_not_divisible_raises(self):
        model = ChunkwiseParallelKDA(key_dim=K, value_dim=V, num_heads=H, chunk_size=8)
        T_bad = 10  # not divisible by 8
        Q = torch.randn(B, T_bad, H, K)
        K_ = torch.randn(B, T_bad, H, K)
        V_ = torch.randn(B, T_bad, H, V)
        g = torch.randn(B, T_bad, H, K)
        beta = torch.rand(B, T_bad, H)
        with pytest.raises(ValueError):
            model(Q, K_, V_, g, beta)

    def test_gradient_flows(self):
        Q, K_, V_, g_log, beta = _small_inputs()
        Q.requires_grad_(True)
        K_.requires_grad_(True)
        V_.requires_grad_(True)
        o, _ = self.model(Q, K_, V_, g_log, beta)
        loss = o.sum()
        loss.backward()
        assert Q.grad is not None
        assert K_.grad is not None

    def test_different_sequence_lengths(self):
        """forward should work for various chunk-divisible T values."""
        for T in [BT, BT * 2, BT * 4]:
            Q = torch.randn(B, T, H, K)
            K_ = F.normalize(torch.randn(B, T, H, K), p=2, dim=-1)
            V_ = torch.randn(B, T, H, V)
            g_log = -torch.rand(B, T, H, K) * 0.5
            beta = torch.rand(B, T, H)
            o, state = self.model(Q, K_, V_, g_log, beta)
            assert o.shape == (B, T, H, V)

    def test_batch_size_1(self):
        T = BT * 2
        Q = torch.randn(1, T, H, K)
        K_ = F.normalize(torch.randn(1, T, H, K), p=2, dim=-1)
        V_ = torch.randn(1, T, H, V)
        g_log = -torch.rand(1, T, H, K) * 0.5
        beta = torch.rand(1, T, H)
        o, state = self.model(Q, K_, V_, g_log, beta)
        assert o.shape == (1, T, H, V)
