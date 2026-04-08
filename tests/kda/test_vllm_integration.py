"""
Tests for src.kda.vllm_integration:
  - KDAStateBlockManager
  - KDAVLLMAdapter
"""
import pytest
import torch
import torch.nn as nn

from src.kda.vllm_integration import KDAStateBlockManager, KDAVLLMAdapter, HAS_VLLM
from src.kda.kda_layer import KDALayer


H_TEST, K_TEST, V_TEST = 4, 16, 16
MAX_BLOCKS = 8
D, NH, DH = 64, 4, 16


@pytest.fixture
def block_manager():
    return KDAStateBlockManager(
        num_heads=H_TEST,
        key_dim=K_TEST,
        value_dim=V_TEST,
        max_blocks=MAX_BLOCKS,
    )


@pytest.fixture
def kda_layer():
    return KDALayer(hidden_dim=D, num_heads=NH, head_dim=DH)


@pytest.fixture
def adapter(kda_layer):
    return KDAVLLMAdapter(
        kda_layer=kda_layer,
        num_heads=NH,
        key_dim=DH,
        value_dim=DH,
        max_blocks=MAX_BLOCKS,
    )


# ── KDAStateBlockManager ──────────────────────────────────────────────────────

class TestKDAStateBlockManager:
    def test_initial_free_count(self, block_manager):
        assert block_manager.num_free_blocks == MAX_BLOCKS

    def test_allocate_returns_correct_shape(self, block_manager):
        state = block_manager.allocate(seq_id=0)
        assert state.shape == (H_TEST, K_TEST, V_TEST)

    def test_allocate_idempotent(self, block_manager):
        s1 = block_manager.allocate(seq_id=1)
        s2 = block_manager.allocate(seq_id=1)
        assert s1.data_ptr() == s2.data_ptr()

    def test_free_increments_pool(self, block_manager):
        block_manager.allocate(seq_id=5)
        before = block_manager.num_free_blocks
        block_manager.free(seq_id=5)
        after = block_manager.num_free_blocks
        assert after == before + 1

    def test_get_returns_none_before_alloc(self, block_manager):
        assert block_manager.get(seq_id=999) is None

    def test_get_returns_block_after_alloc(self, block_manager):
        block_manager.allocate(seq_id=7)
        block = block_manager.get(seq_id=7)
        assert block is not None
        assert block.shape == (H_TEST, K_TEST, V_TEST)

    def test_write_updates_pool(self, block_manager):
        block_manager.allocate(seq_id=2)
        new_state = torch.ones(H_TEST, K_TEST, V_TEST)
        block_manager.write(seq_id=2, state=new_state)
        read_back = block_manager.get(seq_id=2)
        assert torch.allclose(read_back, new_state)

    def test_pool_exhaustion_raises(self):
        small = KDAStateBlockManager(num_heads=H_TEST, key_dim=K_TEST,
                                     value_dim=V_TEST, max_blocks=2)
        small.allocate(seq_id=0)
        small.allocate(seq_id=1)
        with pytest.raises(RuntimeError):
            small.allocate(seq_id=2)

    def test_reuse_after_free(self):
        small = KDAStateBlockManager(num_heads=H_TEST, key_dim=K_TEST,
                                     value_dim=V_TEST, max_blocks=1)
        small.allocate(seq_id=0)
        small.free(seq_id=0)
        small.allocate(seq_id=1)  # should succeed

    def test_block_size_bytes_positive(self, block_manager):
        assert block_manager.block_size_bytes() > 0

    def test_invalid_dimensions_raise(self):
        with pytest.raises(ValueError):
            KDAStateBlockManager(num_heads=0, key_dim=K_TEST, value_dim=V_TEST)


# ── KDAVLLMAdapter ────────────────────────────────────────────────────────────

class TestKDAVLLMAdapterPrefill:
    def test_prefill_output_shape(self, adapter):
        x = torch.randn(2, 16, D)
        output, state = adapter.prefill(x, seq_ids=[0, 1])
        assert output.shape == (2, 16, D)
        assert state.shape == (2, NH, DH, DH)

    def test_prefill_commits_state(self, adapter):
        x = torch.randn(2, 16, D)
        adapter.prefill(x, seq_ids=[10, 11])
        assert adapter._block_manager.get(10) is not None
        assert adapter._block_manager.get(11) is not None

    def test_prefill_no_nan(self, adapter):
        x = torch.randn(2, 16, D)
        output, state = adapter.prefill(x)
        assert not torch.isnan(output).any()
        assert not torch.isnan(state).any()

    def test_prefill_seq_ids_length_mismatch_raises(self, adapter):
        x = torch.randn(2, 16, D)
        with pytest.raises(ValueError):
            adapter.prefill(x, seq_ids=[0])  # B=2 but only 1 seq_id


class TestKDAVLLMAdapterDecodeStep:
    def test_decode_step_output_shape(self, adapter):
        x = torch.randn(2, 16, D)
        adapter.prefill(x, seq_ids=[20, 21])

        x_tok = torch.randn(2, 1, D)
        output, state = adapter.decode_step(x_tok, seq_ids=[20, 21])
        assert output.shape == (2, 1, D)
        assert state.shape == (2, NH, DH, DH)

    def test_decode_step_no_nan(self, adapter):
        x = torch.randn(2, 16, D)
        adapter.prefill(x, seq_ids=[30, 31])
        x_tok = torch.randn(2, 1, D)
        output, _ = adapter.decode_step(x_tok, seq_ids=[30, 31])
        assert not torch.isnan(output).any()

    def test_decode_step_raises_without_prefill(self, adapter):
        x_tok = torch.randn(2, 1, D)
        with pytest.raises(KeyError):
            adapter.decode_step(x_tok, seq_ids=[99, 100])

    def test_decode_step_t_not_1_raises(self, adapter):
        x = torch.randn(2, 16, D)
        adapter.prefill(x, seq_ids=[40, 41])
        x_bad = torch.randn(2, 3, D)  # T=3
        with pytest.raises(ValueError):
            adapter.decode_step(x_bad, seq_ids=[40, 41])

    def test_state_updates_between_steps(self, adapter):
        x = torch.randn(1, 16, D)
        adapter.prefill(x, seq_ids=[50])

        state_before = adapter._block_manager.get(50).clone()
        x_tok = torch.randn(1, 1, D)
        adapter.decode_step(x_tok, seq_ids=[50])
        state_after = adapter._block_manager.get(50).clone()

        # State should change after a decode step
        assert not torch.allclose(state_before, state_after)


class TestKDAVLLMAdapterMisc:
    def test_free_sequence(self, adapter):
        x = torch.randn(1, 8, D)
        adapter.prefill(x, seq_ids=[60])
        assert adapter._block_manager.get(60) is not None
        adapter.free_sequence(60)
        assert adapter._block_manager.get(60) is None

    def test_block_size_bytes_positive(self, adapter):
        assert adapter.get_state_block_size_bytes() > 0

    def test_has_vllm_flag(self):
        assert isinstance(HAS_VLLM, bool)
