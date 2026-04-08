"""
Fine-grained gating mechanism for Kimi Delta Attention (KDA).

Implements channel-wise per-dimension decay gates α_t ∈ (0,1)^(H×K) that
control how much each memory cell forgets at each timestep. This stands in
contrast to coarse head-wise gating (Mamba2, GDN) which uses a single scalar
gate per head, severely limiting the expressiveness of finite-state RNN memory.

Architecture reference: Kimi Linear (arXiv:2510.26692), §3.2 Fine-Grained Gating
"""

# ─────────────────────────────────────────────────────────────────────────────
# MODULE SPEC
# ID:            KDA-GATE-MOD-001
# Requirement:   Provide differentiable per-channel forget gates bounded in (0,1)
#                from an H×D-dimensional input with O(D·rank) parameter cost.
# Purpose:       Enable fine-grained memory management for KDA recurrent state,
#                allowing the model to independently control retention of each
#                feature dimension across all attention heads.
# Rationale:     Head-wise gating (one α per head) collapses all K dimensions
#                into a single forgetting decision, wasting state capacity.
#                Per-channel gating with low-rank bottleneck achieves expressiveness
#                at negligible parameter overhead (rank ≪ H·K).
# Assumptions:   Input is fp32/bf16/fp16, 3-D (B, T, D). Gates are applied
#                element-wise; no inter-head or inter-channel dependencies.
# Constraints:   Gate values strictly in (0,1) via sigmoid — no hard clipping.
# References:    arXiv:2510.26692 §3.2; Gated DeltaNet (Yang et al. 2024)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

logger = logging.getLogger(__name__)


class FineGrainedGating(nn.Module):
    # ─────────────────────────────────────────────────────────────────────────
    # CLASS SPEC
    # ID:            KDA-GATE-CLS-001
    # Requirement:   Given x ∈ R^(B×T×D), return α ∈ (0,1)^(B×T×H×K) using
    #                a low-rank bottleneck W_down∈R^(D×r), W_up∈R^(r×HK).
    # Purpose:       Produce per-timestep, per-channel forget gates for the
    #                KDA state transition: S_t = Diag(α_t) · S_{t-1} + ...
    # Rationale:     Low-rank projection (rank r < H·K) reduces parameters from
    #                D·H·K to D·r + r·H·K while preserving expressiveness.
    #                SiLU activates the bottleneck for smooth gradients.
    #                Sigmoid bounds output strictly in (0,1) for valid decay.
    # Inputs:        hidden_dim ∈ Z+; head_dim ∈ Z+; num_heads ∈ Z+;
    #                rank ∈ Z+ (default: head_dim); dropout ∈ [0,1)
    # Outputs:       gates ∈ (0,1)^(B×T×H×K); optional timing float
    # Preconditions: hidden_dim == x.shape[-1] at forward time
    # Postconditions:All gate values satisfy 0 < α < 1 (sigmoid guarantees)
    # Failure Modes: CUDA OOM → cleared and re-raised; shape mismatch → ValueError
    # Constraints:   Trained with fp32 accumulation recommended for stability
    # Verification:  tests/kda/test_gating.py
    # References:    arXiv:2510.26692 §3.2
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        hidden_dim: int,
        head_dim: int,
        num_heads: int,
        rank: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.rank = rank if rank is not None else head_dim
        self.dropout_p = dropout

        # Low-rank bottleneck: D → r → H·K
        self.gate_down = nn.Linear(hidden_dim, self.rank, bias=False)
        self.gate_up = nn.Linear(self.rank, head_dim * num_heads, bias=False)

        self.drop = nn.Dropout(dropout) if dropout > 0.0 else None

        self._init_weights()

        # Instrumentation
        self._fwd_time_ms: float = 0.0
        self._fwd_calls: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-GATE-INIT-001
    # Requirement:   Initialize gate projections for near-uniform initial gates.
    # Purpose:       Start training with α ≈ 0.5 so gradient flow is symmetric
    #                and neither over-forgetting nor over-remembering initially.
    # Rationale:     Xavier uniform on gate_down gives O(1/√D) variance.
    #                Near-zero gate_up weights push pre-sigmoid activations
    #                toward 0, which maps to sigmoid(0) = 0.5 — uniform gates.
    # Side Effects:  Modifies gate_down.weight and gate_up.weight in-place.
    # Failure Modes: Falls back to default init on any exception (non-fatal).
    # ─────────────────────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        try:
            nn.init.xavier_uniform_(self.gate_down.weight)
            nn.init.uniform_(self.gate_up.weight, -0.01, 0.01)
        except Exception as exc:
            logger.warning(
                "Gate weight initialization failed (%s); using default init.", exc
            )

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-GATE-FWD-001
    # Requirement:   Compute α_t = sigmoid(W_up · SiLU(W_down · x_t)) for each
    #                token position and return tensor with shape (B, T, H, K).
    # Purpose:       Produce per-channel forget gates consumed by DPLRTransition
    #                and StateManager for the KDA state update.
    # Rationale:     SiLU (Swish) activates the bottleneck for smooth, non-zero
    #                gradients. Sigmoid bounds output strictly in (0,1).
    # Inputs:        x ∈ R^(B×T×D); return_timing: bool
    # Outputs:       gates ∈ (0,1)^(B×T×H×K); timing_ms: float | None
    # Preconditions: x.ndim == 3; x.shape[-1] == self.hidden_dim
    # Postconditions:gates.shape == (B, T, num_heads, head_dim)
    #                gates.min() > 0.0; gates.max() < 1.0
    # Side Effects:  Updates _fwd_time_ms and _fwd_calls when return_timing=True
    # Failure Modes: OOM → CUDA cache cleared, RuntimeError re-raised
    #                shape mismatch → ValueError
    # Constraints:   Dropout only active in training mode
    # Verification:  tests/kda/test_gating.py::test_output_shape,
    #                tests/kda/test_gating.py::test_gates_in_unit_range
    # References:    arXiv:2510.26692 Eq. (4)
    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        return_timing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3-D input (B, T, D), got shape {x.shape}"
            )
        B, T, D = x.shape
        if D != self.hidden_dim:
            raise ValueError(
                f"Input last dim {D} != hidden_dim {self.hidden_dim}"
            )

        t0 = time.perf_counter() if return_timing else None

        try:
            h = self.gate_down(x)           # (B, T, rank)
            h = F.silu(h)
            if self.drop is not None and self.training:
                h = self.drop(h)
            gates = self.gate_up(h)         # (B, T, H*K)
            gates = gates.view(B, T, self.num_heads, self.head_dim)
            gates = torch.sigmoid(gates)    # ∈ (0, 1)

            if return_timing:
                elapsed = (time.perf_counter() - t0) * 1_000
                self._fwd_time_ms += elapsed
                self._fwd_calls += 1
                return gates, elapsed

            return gates, None

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.error(
                    "OOM in FineGrainedGating (input %s). Clearing CUDA cache.",
                    x.shape,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise
        except Exception as exc:
            logger.error("FineGrainedGating.forward failed: %s", exc)
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-GATE-METRICS-001
    # Requirement:   Expose average forward pass latency for profiling.
    # Outputs:       float in milliseconds; 0.0 if no calls recorded
    # Side Effects:  None (read-only)
    # ─────────────────────────────────────────────────────────────────────────
    def get_average_time(self) -> float:
        if self._fwd_calls == 0:
            return 0.0
        return self._fwd_time_ms / self._fwd_calls

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-GATE-METRICS-002
    # Requirement:   Reset timing accumulators without affecting model weights.
    # Side Effects:  Sets _fwd_time_ms = 0.0 and _fwd_calls = 0
    # ─────────────────────────────────────────────────────────────────────────
    def reset_timing(self) -> None:
        self._fwd_time_ms = 0.0
        self._fwd_calls = 0

    @property
    def forward_calls(self) -> int:
        return self._fwd_calls

    @property
    def forward_time(self) -> float:
        return self._fwd_time_ms

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, head_dim={self.head_dim}, "
            f"num_heads={self.num_heads}, rank={self.rank}, dropout={self.dropout_p}"
        )
