"""
vLLM-compatible KDA inference adapter.

Provides a PagedAttention-compatible interface for deploying KDA-based models
via vLLM.  Rather than storing per-token K/V pairs, KDA stores a fixed-size
recurrent state S ∈ R^(H × d_k × d_v), which acts as a constant-size KV-cache
block regardless of sequence length.

Key design:
  - KDAStateBlockManager: pre-allocates a pool of state tensors and provides
    alloc/free operations compatible with vLLM's block-allocator interface.
  - KDAVLLMAdapter: wraps a KDALayer and exposes prefill() / decode_step()
    matching the vLLM model runner calling convention.

Reference:
  github.com/vllm-project/vllm — model runner interface
  arXiv:2510.26692 §4 — "each KDA layer maintains a fixed-size state"
"""

# ─────────────────────────────────────────────────────────────────────────────
# MODULE SPEC
# ID:            KDA-VLLM-MOD-001
# Requirement:   Enable vLLM to host KDA-based Kimi Linear models by bridging
#                the recurrent-state KV cache into vLLM's block allocator.
# Purpose:       Production inference deployment with continuous batching.
# Assumptions:   vLLM is not installed; adapter degrades gracefully.
# References:    arXiv:2510.26692 §4; vllm-project/vllm model runner docs
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── vLLM availability probe ───────────────────────────────────────────────────
HAS_VLLM: bool = False
try:
    import vllm  # noqa: F401
    HAS_VLLM = True
    logger.info("KDA-VLLM: vLLM package detected.")
except ImportError:
    logger.info("KDA-VLLM: vLLM not installed — adapter runs in standalone mode.")


# ─────────────────────────────────────────────────────────────────────────────
# CLASS SPEC
# ID:            KDA-VLLM-BLKMGR-001
# Requirement:   Pre-allocate a pool of recurrent-state tensors and manage
#                alloc/free for per-sequence KDA states in O(1) time.
# Purpose:       Replace per-token K/V block storage with fixed-size state.
# Inputs:        num_heads, key_dim, value_dim, max_blocks, dtype
# Failure Modes: Pool exhaustion raises RuntimeError("KDA state pool full")
# Verification:  tests/kda/test_vllm_integration.py::test_block_manager
# ─────────────────────────────────────────────────────────────────────────────
class KDAStateBlockManager:
    """Manages a pool of fixed-size KDA recurrent state tensors."""

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        max_blocks: int = 512,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        if num_heads <= 0 or key_dim <= 0 or value_dim <= 0:
            raise ValueError("num_heads, key_dim, value_dim must be positive.")
        if max_blocks <= 0:
            raise ValueError(f"max_blocks must be positive: {max_blocks}")

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.max_blocks = max_blocks
        self.dtype = dtype
        self.device = device or torch.device("cpu")

        # Pre-allocated state pool: (max_blocks, H, d_k, d_v)
        self._pool = torch.zeros(
            max_blocks, num_heads, key_dim, value_dim,
            dtype=dtype, device=self.device,
        )
        self._free_blocks: List[int] = list(range(max_blocks))
        self._allocated: Dict[int, int] = {}  # seq_id → block_idx

    def allocate(self, seq_id: int) -> torch.Tensor:
        """Allocate a zero-initialised state block for sequence seq_id.

        Returns:
            View of shape (H, d_k, d_v).
        Raises:
            RuntimeError if the pool is exhausted.
        """
        if seq_id in self._allocated:
            return self._pool[self._allocated[seq_id]]

        if not self._free_blocks:
            raise RuntimeError(
                "KDA state pool full — increase max_blocks or free unused sequences."
            )

        block_idx = self._free_blocks.pop()
        self._pool[block_idx].zero_()
        self._allocated[seq_id] = block_idx
        return self._pool[block_idx]

    def free(self, seq_id: int) -> None:
        """Release the state block held by seq_id."""
        block_idx = self._allocated.pop(seq_id, None)
        if block_idx is not None:
            self._free_blocks.append(block_idx)

    def get(self, seq_id: int) -> Optional[torch.Tensor]:
        """Return the current state block for seq_id (None if unallocated)."""
        block_idx = self._allocated.get(seq_id)
        if block_idx is None:
            return None
        return self._pool[block_idx]

    def write(self, seq_id: int, state: torch.Tensor) -> None:
        """Write new state (H, d_k, d_v) back into the pool block."""
        self._pool[self._allocated[seq_id]].copy_(state)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        return len(self._allocated)

    def block_size_bytes(self) -> int:
        """Bytes occupied by one state block."""
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        return self.num_heads * self.key_dim * self.value_dim * element_size

    def extra_repr(self) -> str:
        return (
            f"max_blocks={self.max_blocks}, "
            f"block_shape=({self.num_heads},{self.key_dim},{self.value_dim}), "
            f"block_bytes={self.block_size_bytes()}, "
            f"free={self.num_free_blocks}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLASS SPEC
# ID:            KDA-VLLM-ADAPT-001
# Requirement:   Wrap a KDALayer to expose prefill() and decode_step() methods
#                compatible with vLLM's PagedAttention model-runner convention.
# Purpose:       Enable continuous batching and prefix caching for KDA models.
# Inputs:        kda_layer: KDALayer instance
# Outputs:       prefill → (logits_hidden, states)
#                decode_step → (hidden_out, updated_state)
# Failure Modes: shape mismatch raises ValueError with descriptive message
# Verification:  tests/kda/test_vllm_integration.py
# ─────────────────────────────────────────────────────────────────────────────
class KDAVLLMAdapter(nn.Module):
    """vLLM-compatible inference adapter wrapping a KDALayer."""

    def __init__(
        self,
        kda_layer: nn.Module,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        max_blocks: int = 512,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.kda_layer = kda_layer
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim

        self._block_manager = KDAStateBlockManager(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            max_blocks=max_blocks,
            dtype=dtype,
        )

    @torch.no_grad()
    def prefill(
        self,
        x: torch.Tensor,
        seq_ids: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prefill (encode) a batch of token sequences.

        Args:
            x:       (B, T, D) token embeddings
            seq_ids: optional list of sequence identifiers for state caching;
                     if None, returns states but does not commit to pool

        Returns:
            (output, final_state)
              output:      (B, T, D) contextualised embeddings
              final_state: (B, H, d_k, d_v) — can be stored by caller
        """
        B = x.shape[0]
        output, final_state = self.kda_layer(x, return_state=True)

        if seq_ids is not None:
            if len(seq_ids) != B:
                raise ValueError(
                    f"seq_ids length {len(seq_ids)} != batch size {B}"
                )
            for i, sid in enumerate(seq_ids):
                self._block_manager.allocate(sid)
                self._block_manager.write(sid, final_state[i])

        return output, final_state

    @torch.no_grad()
    def decode_step(
        self,
        x_token: torch.Tensor,
        seq_ids: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-token autoregressive decode step for a batch of sequences.

        Args:
            x_token: (B, 1, D) — current token embeddings
            seq_ids: list of length B — identifies which state blocks to load

        Returns:
            (output, updated_state)
              output:        (B, 1, D) — output embeddings
              updated_state: (B, H, d_k, d_v) — new KDA state (also committed)
        """
        B = x_token.shape[0]
        if x_token.shape[1] != 1:
            raise ValueError(
                f"decode_step expects T=1, got shape {x_token.shape}"
            )
        if len(seq_ids) != B:
            raise ValueError(f"seq_ids length {len(seq_ids)} != batch size {B}")

        # ── Load states from pool ─────────────────────────────────────────
        states = []
        for sid in seq_ids:
            s = self._block_manager.get(sid)
            if s is None:
                raise KeyError(
                    f"Sequence {sid} has no allocated state block. "
                    "Call prefill() or allocate() first."
                )
            states.append(s.unsqueeze(0))
        state_batch = torch.cat(states, dim=0)  # (B, H, d_k, d_v)

        # ── Recurrent decode ──────────────────────────────────────────────
        output, new_state = self.kda_layer(x_token, state=state_batch, return_state=True)

        # ── Write updated states back ─────────────────────────────────────
        for i, sid in enumerate(seq_ids):
            self._block_manager.write(sid, new_state[i])

        return output, new_state

    def free_sequence(self, seq_id: int) -> None:
        """Release the state block for a finished sequence."""
        self._block_manager.free(seq_id)

    def get_state_block_size_bytes(self) -> int:
        """Bytes per state block — for vLLM block allocator sizing."""
        return self._block_manager.block_size_bytes()

    @property
    def num_free_blocks(self) -> int:
        return self._block_manager.num_free_blocks

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, key_dim={self.key_dim}, "
            f"value_dim={self.value_dim}, has_vllm={HAS_VLLM}, "
            f"block_bytes={self.get_state_block_size_bytes()}"
        )
