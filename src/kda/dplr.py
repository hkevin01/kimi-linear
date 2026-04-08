"""
Diagonal-Plus-Low-Rank (DPLR) state transition for Kimi Delta Attention.

Implements the constrained KDA variant of the DPLR transition matrix, which
reduces the general O(K²·V) DPLR to O(K·V) by exploiting the structural
constraint at = βt kt, bt = kt ⊙ αt. This constraint means the rank-1
correction and the diagonal decay share the same key vector kt, enabling
the two-step decomposition: diagonal decay followed by a rank-1 correction.

Architecture reference: Kimi Linear (arXiv:2510.26692), §3.3 DPLR Transition
"""

# ─────────────────────────────────────────────────────────────────────────────
# MODULE SPEC
# ID:            KDA-DPLR-MOD-001
# Requirement:   Implement the KDA state transition S_{t} = A_t S_{t-1} + β_t k_t v_t^T
#                where A_t = Diag(α_t) − β_t k_t k_t^T Diag(α_t).
# Purpose:       Update the finite-state RNN memory matrix S ∈ R^(K×V) at each
#                token position, combining diagonal forgetting with delta-rule
#                error correction and new key-value association.
# Rationale:     The constrained DPLR factorisation allows the full update to be
#                computed in two matrix-vector products instead of one full
#                matrix-matrix product, halving the dominant FLOP count.
# Assumptions:   Keys are L2-normalised before calling compute_transition.
#                β_t values are bounded in (0,1) by upstream sigmoid.
# Constraints:   K = V = head_dim in the default KDA configuration.
# References:    arXiv:2510.26692 §3.3; DeltaNet (Schlag et al. 2021)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

logger = logging.getLogger(__name__)


class DPLRTransition(nn.Module):
    # ─────────────────────────────────────────────────────────────────────────
    # CLASS SPEC
    # ID:            KDA-DPLR-CLS-001
    # Requirement:   Given state S, normalised key k, value v, gate α and scalar
    #                β, return the updated state S' = A·S + β·k·v^T where
    #                A = Diag(α) − β·k·k^T·Diag(α).
    # Purpose:       Encapsulate the KDA recurrent state update with numerical
    #                stability monitoring and performance instrumentation.
    # Rationale:     Separating transition from the full KDALayer allows the
    #                operation to be independently tested, profiled, and later
    #                replaced with a hardware kernel (e.g., Triton) without
    #                changing downstream code.
    # Inputs:        key_dim ∈ Z+; value_dim ∈ Z+; num_heads ∈ Z+;
    #                use_eigenvalue_stabilization: bool (default True)
    # Outputs:       Updated state tensor ∈ R^(B×H×K×V)
    # Failure Modes: NaN in output → returns state_decayed (fallback);
    #                Inf in output → clamps to ±1e6; OOM → cleared and re-raised
    # Verification:  tests/kda/test_dplr.py
    # References:    arXiv:2510.26692 §3.3 Eq. (5)
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        num_heads: int,
        use_eigenvalue_stabilization: bool = True,
    ) -> None:
        super().__init__()

        if key_dim <= 0 or value_dim <= 0:
            raise ValueError(
                f"Dimensions must be positive: key_dim={key_dim}, value_dim={value_dim}"
            )
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive: {num_heads}")

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.use_eigenvalue_stabilization = use_eigenvalue_stabilization

        # Instrumentation
        self._fwd_time_ms: float = 0.0
        self._fwd_calls: int = 0
        self._eigenvalue_warnings: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-DPLR-TRANS-001
    # Requirement:   Compute S' = (Diag(α) − β·k·k^T·Diag(α)) · S
    #                in two sequential steps with no materialised K×K matrix.
    # Purpose:       Apply diagonal forgetting plus rank-1 delta correction to
    #                the current memory state before adding the new association.
    # Rationale:     Factorising into step1 (diagonal decay) then step2 (rank-1
    #                correction) reduces flops: O(B·H·K·V) vs O(B·H·K²·V).
    #                Step1: S' = Diag(α)·S  via element-wise broadcast.
    #                Step2: S'' = S' − β·k·(k^T·S')  via two einsum calls.
    # Inputs:        state ∈ R^(B×H×K×V); keys ∈ R^(B×H×K) L2-normalised;
    #                gates ∈ (0,1)^(B×H×K); beta ∈ (0,1)^(B×H×1);
    #                return_timing: bool
    # Outputs:       transitioned_state ∈ R^(B×H×K×V); timing_ms: float | None
    # Preconditions: keys must be L2-normalised (‖k‖₂ = 1 per head)
    #                gates ∈ (0,1); beta ∈ (0,1)
    # Postconditions:output shape == input state shape
    #                No NaN values (fallback to state_decayed if NaN detected)
    # Side Effects:  _eigenvalue_warnings incremented if instability detected
    # Failure Modes: NaN → logs warning, returns state_decayed
    #                Inf → logs warning, clamps to [-1e6, 1e6]
    #                OOM → logs error, clears CUDA cache, re-raises
    # Constraints:   keys expected to be L2-normalised before entry
    # Verification:  tests/kda/test_dplr.py::test_transition_shape,
    #                tests/kda/test_dplr.py::test_heavy_forgetting
    # References:    arXiv:2510.26692 §3.3 Eq. (5)
    # ─────────────────────────────────────────────────────────────────────────
    def compute_transition(
        self,
        state: torch.Tensor,
        keys: torch.Tensor,
        gates: torch.Tensor,
        beta: torch.Tensor,
        return_timing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        t0 = time.perf_counter() if return_timing else None

        B, H, K, V = state.shape
        if keys.shape != (B, H, K):
            raise ValueError(
                f"keys shape {keys.shape} != expected {(B, H, K)}"
            )
        if gates.shape != (B, H, K):
            raise ValueError(
                f"gates shape {gates.shape} != expected {(B, H, K)}"
            )
        if beta.shape[:-1] != (B, H):
            raise ValueError(
                f"beta leading dims {beta.shape[:-1]} != expected {(B, H)}"
            )

        try:
            # ── Step 1: diagonal decay ─────────────────────────────────────
            # S' = Diag(α_t) · S_{t-1}  →  broadcast (B,H,K,1) * (B,H,K,V)
            state_decayed = gates.unsqueeze(-1) * state  # (B, H, K, V)

            if self.use_eigenvalue_stabilization:
                self._check_eigenvalue_stability(gates, keys, beta)

            # ── Step 2: rank-1 delta correction ───────────────────────────
            # S'' = (I − β·k·k^T) · S'
            # Efficient: S'' = S' − β·k·(k^T·S')
            beta_exp = beta.unsqueeze(-1)  # (B, H, 1, 1)
            kt_S = torch.einsum("bhk,bhkv->bhv", keys, state_decayed)     # (B, H, V)
            correction = beta_exp * torch.einsum("bhk,bhv->bhkv", keys, kt_S)  # (B, H, K, V)
            transitioned = state_decayed - correction

            # ── Numerical guard ────────────────────────────────────────────
            if torch.isnan(transitioned).any():
                logger.warning(
                    "NaN in DPLR transition (B=%d H=%d K=%d V=%d); "
                    "falling back to decayed state.",
                    B, H, K, V,
                )
                transitioned = state_decayed
            elif torch.isinf(transitioned).any():
                logger.warning("Inf in DPLR transition; clamping to ±1e6.")
                transitioned = transitioned.clamp(-1e6, 1e6)

            if return_timing:
                elapsed = (time.perf_counter() - t0) * 1_000
                self._fwd_time_ms += elapsed
                self._fwd_calls += 1
                return transitioned, elapsed

            return transitioned, None

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.error(
                    "OOM in DPLRTransition.compute_transition (state %s).",
                    state.shape,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise
        except Exception as exc:
            logger.error("DPLRTransition.compute_transition failed: %s", exc)
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-DPLR-STAB-001
    # Requirement:   Emit a warning when the spectral radius of A_t is estimated
    #                to exceed the stability threshold (default 1.1).
    # Purpose:       Detect potential divergence early during training without
    #                paying the cost of a full eigen-decomposition.
    # Rationale:     For L2-normalised keys (‖k‖=1) the Gershgorin circle
    #                heuristic bounds the dominant eigenvalue by
    #                max(α) · (1 + β·‖k‖²).  When this exceeds the threshold,
    #                weight decay or β regularisation should be checked.
    # Inputs:        gates ∈ (0,1)^(B×H×K); keys ∈ R^(B×H×K);
    #                beta ∈ (0,1)^(B×H×1); threshold: float
    # Side Effects:  _eigenvalue_warnings count incremented; logger.warning emitted
    #                (at most 3 times total to prevent log flooding)
    # Failure Modes: Any exception is silently caught — stability check must
    #                never interrupt the forward pass.
    # References:    Gershgorin circle theorem; arXiv:2510.26692 §A.1
    # ─────────────────────────────────────────────────────────────────────────
    def _check_eigenvalue_stability(
        self,
        gates: torch.Tensor,
        keys: torch.Tensor,
        beta: torch.Tensor,
        threshold: float = 1.1,
    ) -> None:
        try:
            max_alpha = gates.max().item()
            # For L2-normalised keys ‖k‖² = 1, so correction term = max(β)
            max_beta = beta.max().item()
            spectral_estimate = max_alpha * (1.0 + max_beta)

            if spectral_estimate > threshold:
                self._eigenvalue_warnings += 1
                if self._eigenvalue_warnings <= 3:
                    logger.warning(
                        "Spectral radius estimate %.4f > %.4f. "
                        "Consider reducing β or tightening key normalisation. "
                        "(warning %d/3)",
                        spectral_estimate,
                        threshold,
                        self._eigenvalue_warnings,
                    )
        except Exception:
            pass  # Stability check is non-critical; never break the forward pass

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-DPLR-FWD-001
    # Requirement:   Compute the full KDA state update:
    #                S_t = A_t · S_{t-1} + β_t · k_t · v_t^T
    #                where A_t is the DPLR transition from compute_transition.
    # Purpose:       Single entry-point that combines transition and KV write for
    #                use in recurrent token-by-token inference or layer stacking.
    # Inputs:        state, keys, values, gates, beta as above; return_timing: bool
    # Outputs:       new_state ∈ R^(B×H×K×V); timing_ms: float | None
    # Preconditions: keys L2-normalised; gates ∈ (0,1); beta ∈ (0,1)
    # Postconditions:new_state.shape == state.shape
    # Side Effects:  _fwd_time_ms and _fwd_calls updated when return_timing=True
    # Verification:  tests/kda/test_dplr.py::test_full_forward_shape
    # References:    arXiv:2510.26692 Eq. (3)
    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        state: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        gates: torch.Tensor,
        beta: torch.Tensor,
        return_timing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        t0 = time.perf_counter() if return_timing else None

        transitioned, _ = self.compute_transition(
            state, keys, gates, beta, return_timing=False
        )

        # KV write: β_t · k_t · v_t^T  →  (B, H, K, V)
        beta_exp = beta.unsqueeze(-1)  # (B, H, 1, 1)
        kv = beta_exp * torch.einsum("bhk,bhv->bhkv", keys, values)
        new_state = transitioned + kv

        if return_timing:
            elapsed = (time.perf_counter() - t0) * 1_000
            self._fwd_time_ms += elapsed
            self._fwd_calls += 1
            return new_state, elapsed

        return new_state, None

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-DPLR-METRICS-001
    # Requirement:   Return mean forward-pass latency in milliseconds.
    # Outputs:       float ≥ 0; 0.0 if no calls have been made
    # ─────────────────────────────────────────────────────────────────────────
    def get_average_time(self) -> float:
        if self._fwd_calls == 0:
            return 0.0
        return self._fwd_time_ms / self._fwd_calls

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-DPLR-METRICS-002
    # Requirement:   Reset all timing and warning counters to zero.
    # Side Effects:  Clears _fwd_time_ms, _fwd_calls, _eigenvalue_warnings
    # ─────────────────────────────────────────────────────────────────────────
    def reset_timing(self) -> None:
        self._fwd_time_ms = 0.0
        self._fwd_calls = 0
        self._eigenvalue_warnings = 0

    @property
    def forward_calls(self) -> int:
        return self._fwd_calls

    @property
    def forward_time(self) -> float:
        return self._fwd_time_ms

    @property
    def eigenvalue_warnings(self) -> int:
        return self._eigenvalue_warnings

    def extra_repr(self) -> str:
        return (
            f"key_dim={self.key_dim}, value_dim={self.value_dim}, "
            f"num_heads={self.num_heads}, "
            f"use_eigenvalue_stabilization={self.use_eigenvalue_stabilization}"
        )
