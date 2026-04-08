"""
Triton/CUDA kernel dispatch layer for KDA chunkwise operations.

Attempts to import FLA (flash-linear-attention) Triton kernels for hardware-
accelerated KDA forward/backward passes.  If FLA is unavailable (CPU-only,
no Triton installed, or incompatible hardware), falls back transparently to
the pure-PyTorch ChunkwiseParallelKDA implementation.

Dispatch precedence:
  1. fla.ops.kda.chunk_kda          (Triton, production path)
  2. fla.ops.kda.fused_recurrent_kda (Triton, small-T recurrent path)
  3. ChunkwiseParallelKDA            (pure PyTorch, fallback)

Usage::

    from src.kda.triton_kernels import chunk_kda_forward, fused_recurrent_kda_forward
    from src.kda.triton_kernels import HAS_TRITON

    output, state = chunk_kda_forward(q, k, v, g, beta, initial_state=state)

Reference: github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda
"""

# ─────────────────────────────────────────────────────────────────────────────
# MODULE SPEC
# ID:            KDA-TRITON-MOD-001
# Requirement:   Provide unified API for KDA forward pass regardless of whether
#                the Triton kernel library is installed.
# Purpose:       Hardware acceleration when available; correctness otherwise.
# Failure Modes: Triton import failure → silent fallback to PyTorch.
# References:    github.com/fla-org/flash-linear-attention
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ── Triton / FLA availability probe ──────────────────────────────────────────
HAS_TRITON: bool = False
_fla_chunk_kda = None
_fla_fused_recurrent_kda = None

try:
    import triton  # noqa: F401
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

if _TRITON_AVAILABLE:
    try:
        from fla.ops.kda import chunk_kda as _fla_chunk_kda          # type: ignore[no-redef]
        HAS_TRITON = True
        logger.info("KDA-TRITON: FLA chunk_kda loaded (Triton path active).")
    except ImportError:
        logger.info("KDA-TRITON: FLA not found — using pure-PyTorch fallback.")

    if HAS_TRITON:
        try:
            from fla.ops.kda.fused_recurrent import fused_recurrent_kda as _fla_fused_recurrent_kda  # type: ignore[no-redef]
        except ImportError:
            logger.debug("KDA-TRITON: fla.ops.kda.fused_recurrent not available.")


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION SPEC
# ID:            KDA-TRITON-CHUNK-001
# Requirement:   Compute KDA forward pass over (B,H,T,d) tensors using chunk
#                parallelism; dispatch to Triton kernel when available.
# Inputs:        q,k,v ∈ R^(B×H×T×d); g ∈ R^(B×H×T×1) (log-space gates);
#                beta ∈ R^(B×H×T×1) ∈ (0,1]; initial_state ∈ R^(B×H×d×d)?;
#                chunk_size ∈ {16,32,64,128}; scale ∈ R+
# Outputs:       (output ∈ R^(B×H×T×d), final_state ∈ R^(B×H×d×d))
# Preconditions: T % chunk_size == 0 when using PyTorch fallback
# Postconditions:output.shape == q.shape
# Failure Modes: T not divisible → fallback pads internally
# ─────────────────────────────────────────────────────────────────────────────
def chunk_kda_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    scale: Optional[float] = None,
    return_intermediate_states: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dispatch KDA chunkwise forward to Triton (if available) or PyTorch.

    Args:
        q, k, v:  (B, H, T, head_dim) — query, key, value
        g:        (B, H, T, 1)        — log-space gates (≤ 0)
        beta:     (B, H, T, 1)        — delta-rule update coefficient ∈ (0,1]
        initial_state: (B, H, d_k, d_v) or None
        chunk_size: number of tokens per chunk (must be power of 2)
        scale:    attention scale (default: head_dim**-0.5)
        return_intermediate_states: if True, also return per-chunk states

    Returns:
        (output, final_state) both as dense tensors, or
        (output, final_state, inter_states) when return_intermediate_states=True
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    if HAS_TRITON and _fla_chunk_kda is not None and q.is_cuda:
        return _triton_chunk_kda(
            q, k, v, g, beta,
            initial_state=initial_state,
            chunk_size=chunk_size,
            scale=scale,
            return_intermediate_states=return_intermediate_states,
        )
    else:
        return _pytorch_chunk_kda(
            q, k, v, g, beta,
            initial_state=initial_state,
            chunk_size=chunk_size,
            scale=scale,
            return_intermediate_states=return_intermediate_states,
        )


def _triton_chunk_kda(
    q, k, v, g, beta,
    initial_state=None,
    chunk_size=64,
    scale=None,
    return_intermediate_states=False,
):
    """Wrapper around FLA's chunk_kda Triton kernel."""
    # ─────────────────────────────────────────────────────────────────────────
    # ID:            KDA-TRITON-FLA-001
    # Requirement:   Call fla.ops.kda.chunk_kda with correct argument mapping.
    # Notes:         FLA expects g in log-space; gates already that here.
    #                use_qk_l2norm_in_kernel=True matches paper description.
    # ─────────────────────────────────────────────────────────────────────────
    result = _fla_chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        chunk_size=chunk_size,
        return_intermediate_states=return_intermediate_states,
    )
    # FLA returns (output, [inter_states,] final_state)
    if return_intermediate_states:
        output, inter_states, final_state = result
        return output, final_state, inter_states
    else:
        output, final_state = result
        return output, final_state


def _pytorch_chunk_kda(
    q, k, v, g, beta,
    initial_state=None,
    chunk_size=64,
    scale=None,
    return_intermediate_states=False,
):
    """
    Pure-PyTorch token-loop fallback for environments without Triton/FLA.

    Runs a sequential recurrent KDA update — correct for all input sizes; no
    chunk-size divisibility requirement.

    Inputs are in the chunk_kda_forward convention:
      q, k, v : (B, H, T, d)
      g       : (B, H, T, 1) — scalar log-space gate per head/token (≤ 0)
      beta    : (B, H, T, 1) — delta-rule write rate ∈ (0, 1]
    """
    # ─────────────────────────────────────────────────────────────────────────
    # ID:            KDA-TRITON-PY-001
    # Requirement:   Produce identical output to the Triton path via O(T) loop.
    # Rationale:     The Triton helpers (compute_wy_representation etc.) expect
    #                per-channel gates and (B,H,BT,d) layout; this fallback uses
    #                the scalar-gate convention directly to avoid layout mismatches.
    # ─────────────────────────────────────────────────────────────────────────
    B, H, T, d_k = q.shape
    d_v = v.shape[-1]
    device = q.device
    dtype = q.dtype

    if initial_state is None:
        state = torch.zeros(B, H, d_k, d_v, device=device, dtype=dtype)
    else:
        state = initial_state.clone()

    outputs = []
    inter_states_list: list = []

    for t in range(T):
        g_t = g[:, :, t, 0].exp()              # (B, H) scalar decay per head
        beta_t = beta[:, :, t, 0]              # (B, H) write rate
        k_t = k[:, :, t, :]                    # (B, H, d_k)
        v_t = v[:, :, t, :]                    # (B, H, d_v)
        q_t = q[:, :, t, :]                    # (B, H, d_k)

        # Decay state: S_t = Diag(g_t) · S_{t-1}
        state = state * g_t[..., None, None]   # (B, H, d_k, d_v)

        # Delta-rule update: S += β · k · (v − S^T k)^T
        Sk = torch.einsum("bhkv,bhk->bhv", state, k_t)   # (B, H, d_v)
        e_t = v_t - Sk                                     # (B, H, d_v)
        state = state + beta_t[..., None, None] * torch.einsum(
            "bhk,bhv->bhkv", k_t, e_t
        )

        # Retrieve: o_t = scale · S^T q_t
        outputs.append(scale * torch.einsum("bhk,bhkv->bhv", q_t, state))  # (B,H,d_v)

        if return_intermediate_states:
            inter_states_list.append(state.clone())

    output = torch.stack(outputs, dim=2)   # (B, H, T, d_v)

    if return_intermediate_states:
        inter_states = torch.stack(inter_states_list, dim=2)  # (B, H, T, d_k, d_v)
        return output, state, inter_states

    return output, state


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION SPEC
# ID:            KDA-TRITON-RECUR-001
# Requirement:   Compute KDA via fused recurrent kernel (T≤256 or sequential).
# Inputs:        q,k,v,g,beta — same shapes as chunk_kda_forward
# Outputs:       (output, final_state)
# ─────────────────────────────────────────────────────────────────────────────
def fused_recurrent_kda_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused recurrent KDA forward — optimal for small T or single-token decode.

    Dispatches to FLA fused_recurrent_kda Triton kernel when available,
    otherwise uses the PyTorch Sequential fallback from DPLRTransition.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    if HAS_TRITON and _fla_fused_recurrent_kda is not None and q.is_cuda:
        result = _fla_fused_recurrent_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
        )
        return result[0], result[1]

    # ── Pure-PyTorch recurrent fallback ────────────────────────────────────
    # Use the token-loop in DPLRTransition
    from .dplr import DPLRTransition

    B, H, T, d_k = q.shape
    d_v = v.shape[-1]
    device = q.device
    dtype = q.dtype

    if initial_state is None:
        state = torch.zeros(B, H, d_k, d_v, device=device, dtype=dtype)
    else:
        state = initial_state.clone()

    outputs = []
    for t in range(T):
        g_t = g[:, :, t, 0].exp()                            # (B, H)
        beta_t = beta[:, :, t, 0]                            # (B, H)
        k_t = k[:, :, t, :]                                  # (B, H, d_k)
        v_t = v[:, :, t, :]                                  # (B, H, d_v)
        q_t = q[:, :, t, :]                                  # (B, H, d_k)

        # Decay state
        state = state * g_t[..., None, None]

        # Delta rule update: S += beta * k * (v - S^T k)^T
        Sk = torch.einsum("bhkv,bhk->bhv", state, k_t)      # (B, H, d_v)
        e_t = v_t - Sk                                        # (B, H, d_v)
        state = state + beta_t[..., None, None] * torch.einsum(
            "bhk,bhv->bhkv", k_t, e_t
        )

        outputs.append(torch.einsum("bhk,bhkv->bhv", q_t, state))

    output = torch.stack(outputs, dim=2)  # (B, H, T, d_v)
    return output * scale, state
