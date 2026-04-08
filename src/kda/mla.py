"""
Multi-Head Latent Attention (MLA) reference module.

MLA reduces the KV-cache footprint versus standard MHA by projecting keys and
values through a shared low-rank "latent" vector c_t ∈ R^(d_c) before
expanding into the per-head K/V spaces.  This is the full-attention component
used in the 3:1 KDA-to-MLA hybrid deployment described in arXiv:2510.26692 §4.

Architecture:
  Q = W_Q · x              (B, T, H·d_q)
  c_kv = W_down_kv · x     (B, T, d_c)          ← shared KV compression
  K = W_up_k · c_kv        (B, T, H·d_k)
  V = W_up_v · c_kv        (B, T, H·d_v)
  Attn = softmax(Q K^T / √d_k) V
  y = W_O · Attn            (B, T, D)

The KV cache at inference stores only c_kv (shape T × d_c) instead of
(T × H·(d_k + d_v)), achieving up to (H·(d_k + d_v)) / d_c compression.

Reference: DeepSeek-V2 (arXiv:2405.04434) §2.1; Kimi Linear §4 hybrid stack.
"""

# ─────────────────────────────────────────────────────────────────────────────
# MODULE SPEC
# ID:            KDA-MLA-MOD-001
# Requirement:   Implement MLA full-attention layer with compressed KV-cache.
# Purpose:       Provide the "global" attention layer in the 3:1 hybrid stack.
# References:    arXiv:2510.26692 §4; arXiv:2405.04434 §2.1
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import math
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MLALayer(nn.Module):
    # ─────────────────────────────────────────────────────────────────────────
    # CLASS SPEC
    # ID:            KDA-MLA-CLS-001
    # Requirement:   Compute scaled dot-product attention with low-rank KV
    #                projection: Q=W_Q·x; c=W_down·x; K=W_upK·c; V=W_upV·c.
    # Purpose:       Full-attention component in the KDA/MLA hybrid block.
    # Inputs:        hidden_dim ∈ Z+; num_heads ∈ Z+; head_dim ∈ Z+;
    #                kv_latent_dim ∈ Z+ (d_c, default hidden_dim // 4);
    #                dropout ∈ [0,1)
    # Outputs:       tensor ∈ R^(B×T×D)
    # Failure Modes: OOM → clear CUDA cache, re-raise
    # Verification:  tests/kda/test_mla.py
    # References:    arXiv:2405.04434 §2.1
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        kv_latent_dim: int = 0,  # 0 → hidden_dim // 4
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError(
                f"Dimensions must be positive: hidden={hidden_dim}, "
                f"heads={num_heads}, head_dim={head_dim}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1): {dropout}")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.kv_latent_dim = kv_latent_dim if kv_latent_dim > 0 else max(1, hidden_dim // 4)
        self.scale = head_dim ** -0.5

        # ── Query projection (full, not compressed) ────────────────────────
        self.q_proj = nn.Linear(hidden_dim, self.inner_dim, bias=False)

        # ── Shared KV compression ──────────────────────────────────────────
        self.kv_down_proj = nn.Linear(hidden_dim, self.kv_latent_dim, bias=False)
        self.k_up_proj = nn.Linear(self.kv_latent_dim, self.inner_dim, bias=False)
        self.v_up_proj = nn.Linear(self.kv_latent_dim, self.inner_dim, bias=False)

        # ── Output projection ──────────────────────────────────────────────
        self.out_proj = nn.Linear(self.inner_dim, hidden_dim, bias=False)

        # ── Attention dropout ──────────────────────────────────────────────
        self.attn_drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Instrumentation
        self._fwd_time_ms: float = 0.0
        self._fwd_calls: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-MLA-FWD-001
    # Requirement:   Compute MLA output:
    #                  Q  = W_Q · x                      (B, T, H*d_k)
    #                  c  = W_down · x                   (B, T, d_c)
    #                  K  = W_upK · c                    (B, T, H*d_k)
    #                  V  = W_upV · c                    (B, T, H*d_v)
    #                  A  = softmax(Q K^T / √d_k)        (B, H, T, T) causal
    #                  y  = W_O · reshape(A V)            (B, T, D)
    # Inputs:        x ∈ R^(B×T×D); kv_cache: optional (c_past, T_past)
    # Outputs:       output ∈ R^(B×T×D);
    #                c_kv ∈ R^(B×T×d_c) (for generation KV cache)
    # Preconditions: x already layer-normed
    # Postconditions:output.shape == x.shape
    # Failure Modes: For T>8192 on small VRAM, may OOM — use chunked attn
    # Verification:  tests/kda/test_mla.py::test_mla_output_shape
    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        return_kv_cache: bool = True,
        return_timing: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        t0 = time.perf_counter() if return_timing else None
        B, T, D = x.shape

        try:
            Q = self.q_proj(x)                          # (B, T, H*d_k)
            c_kv = self.kv_down_proj(x)                 # (B, T, d_c)

            if kv_cache is not None:
                # Concatenate past latent vectors for generation
                c_full = torch.cat([kv_cache, c_kv], dim=1)
            else:
                c_full = c_kv                           # (B, T_full, d_c)

            K = self.k_up_proj(c_full)                  # (B, T_full, H*d_k)
            V = self.v_up_proj(c_full)                  # (B, T_full, H*d_v)

            T_full = K.shape[1]

            # Reshape to multi-head: (B, H, T, d)
            Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(B, T_full, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(B, T_full, self.num_heads, self.head_dim).transpose(1, 2)

            # ── Causal scaled dot-product attention ────────────────────────
            # Use PyTorch's efficient SDPA (uses Flash Attention where available)
            attn_out = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=0.0,
                is_causal=(kv_cache is None),  # not strictly causal if appending cache
            )                                            # (B, H, T, d_v)

            # Merge heads
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.inner_dim)
            output = self.out_proj(attn_out)

            result: list = [output]
            if return_kv_cache:
                result.append(c_kv)
            if return_timing:
                elapsed = (time.perf_counter() - t0) * 1_000
                self._fwd_time_ms += elapsed
                self._fwd_calls += 1
                result.append(elapsed)

            return tuple(result) if len(result) > 1 else output

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.error("OOM in MLALayer.forward (x %s).", x.shape)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise

    def get_average_time(self) -> float:
        if self._fwd_calls == 0:
            return 0.0
        return self._fwd_time_ms / self._fwd_calls

    def reset_timing(self) -> None:
        self._fwd_time_ms = 0.0
        self._fwd_calls = 0

    @property
    def kv_cache_compression_ratio(self) -> float:
        """Ratio of standard KV cache size to MLA compressed KV cache size."""
        standard = self.num_heads * (self.head_dim + self.head_dim)  # K + V per head
        return standard / self.kv_latent_dim

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, kv_latent_dim={self.kv_latent_dim}, "
            f"compression_ratio={self.kv_cache_compression_ratio:.1f}x"
        )
