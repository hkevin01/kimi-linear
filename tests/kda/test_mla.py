"""
Tests for src.kda.mla.MLALayer
"""
import pytest
import torch
import torch.nn.functional as F

from src.kda.mla import MLALayer


B, T, D = 2, 16, 64
H, DH = 4, 16


@pytest.fixture
def model():
    return MLALayer(hidden_dim=D, num_heads=H, head_dim=DH, kv_latent_dim=16)


class TestMLALayerConstruction:
    def test_default_latent_dim(self):
        m = MLALayer(hidden_dim=D, num_heads=H, head_dim=DH)
        assert m.kv_latent_dim == max(1, D // 4)

    def test_custom_latent_dim(self):
        m = MLALayer(hidden_dim=D, num_heads=H, head_dim=DH, kv_latent_dim=8)
        assert m.kv_latent_dim == 8

    def test_invalid_dims_raise(self):
        with pytest.raises(ValueError):
            MLALayer(hidden_dim=0, num_heads=H, head_dim=DH)
        with pytest.raises(ValueError):
            MLALayer(hidden_dim=D, num_heads=0, head_dim=DH)

    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError):
            MLALayer(hidden_dim=D, num_heads=H, head_dim=DH, dropout=1.0)

    def test_compression_ratio_positive(self):
        m = MLALayer(hidden_dim=D, num_heads=H, head_dim=DH, kv_latent_dim=8)
        assert m.kv_cache_compression_ratio > 0


class TestMLALayerForward:
    def test_output_shape(self, model):
        x = torch.randn(B, T, D)
        result = model(x)
        # single output (no kv_cache returned when return_kv_cache=False)
        output = model(x, return_kv_cache=False)
        assert output.shape == (B, T, D)

    def test_output_shape_with_kv_cache(self, model):
        x = torch.randn(B, T, D)
        output, c_kv = model(x, return_kv_cache=True)
        assert output.shape == (B, T, D)
        assert c_kv.shape == (B, T, model.kv_latent_dim)

    def test_no_nan(self, model):
        x = torch.randn(B, T, D)
        output, c_kv = model(x)
        assert not torch.isnan(output).any(), "NaN in output"
        assert not torch.isnan(c_kv).any(), "NaN in c_kv"

    def test_no_inf(self, model):
        x = torch.randn(B, T, D)
        output, _ = model(x)
        assert not torch.isinf(output).any()

    def test_kv_cache_passthrough(self, model):
        """Passing kv_cache from step-1 should produce T=1 decode output."""
        x_prefill = torch.randn(B, T, D)
        _, c_kv_past = model(x_prefill)

        x_decode = torch.randn(B, 1, D)
        output, c_kv_new = model(x_decode, kv_cache=c_kv_past)
        assert output.shape == (B, 1, D)
        assert c_kv_new.shape == (B, 1, model.kv_latent_dim)

    def test_gradient_flows(self, model):
        x = torch.randn(B, T, D, requires_grad=True)
        output, _ = model(x)
        output.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_batch_size_1(self, model):
        x = torch.randn(1, T, D)
        output, c_kv = model(x)
        assert output.shape == (1, T, D)

    def test_timing_instrumentation(self, model):
        x = torch.randn(B, T, D)
        model.reset_timing()
        model(x, return_kv_cache=False, return_timing=True)
        assert model._fwd_calls == 1
        assert model.get_average_time() > 0


class TestMLALayerParameterCount:
    def test_has_q_proj(self, model):
        assert hasattr(model, "q_proj")

    def test_has_kv_projections(self, model):
        assert hasattr(model, "kv_down_proj")
        assert hasattr(model, "k_up_proj")
        assert hasattr(model, "v_up_proj")

    def test_has_out_proj(self, model):
        assert hasattr(model, "out_proj")
