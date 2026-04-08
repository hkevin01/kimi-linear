"""
Recurrent state management for Kimi Delta Attention (KDA).

Maintains the matrix-valued state S_t ∈ R^(B×H×K×V) that accumulates
key-value associations with forgetting. The state has constant memory
O(B·H·K·V) regardless of sequence length, giving KDA its linear-time
inference property over unbounded contexts.

Architecture reference: Kimi Linear (arXiv:2510.26692), §3
"""

# ─────────────────────────────────────────────────────────────────────────────
# MODULE SPEC
# ID:            KDA-SM-MOD-001
# Requirement:   Store, initialise, update, and checkpoint the KDA state tensor
#                S_t ∈ R^(B×H×K×V) across token positions.
# Purpose:       Centralise all stateful memory management for KDA so that the
#                higher-level KDALayer can focus on attention computation.
# Rationale:     Separating state management from attention logic simplifies
#                debugging, checkpointing during long-context inference, and
#                future replacement with custom CUDA state kernels.
# Assumptions:   B ≤ max_batch_size; keys are L2-normalised by caller;
#                gates ∈ (0,1); beta ∈ (0,1).
# Constraints:   Memory footprint: max_batch_size × num_heads × K × V × dtype bytes
# References:    arXiv:2510.26692 §3 Eq. (3)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import warnings
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class StateManager(nn.Module):
    # ─────────────────────────────────────────────────────────────────────────
    # CLASS SPEC
    # ID:            KDA-SM-CLS-001
    # Requirement:   Provide initialize_state / update_state / checkpoint APIs
    #                with shape validation, NaN guards, OOM recovery, and timing.
    # Purpose:       Single class responsible for the lifecycle of S_t from
    #                construction through sequential update to persistence.
    # Rationale:     Pre-allocating a fixed state_buffer avoids repeated host-to-
    #                device copies during inference; cloning when returning keeps
    #                internal buffer independent of caller mutations.
    # Inputs:        key_dim ∈ Z+; value_dim ∈ Z+; num_heads ∈ Z+;
    #                max_batch_size ∈ Z+; dtype: torch.dtype; device: str
    # Outputs:       Validated state tensors; memory stats; timing metrics
    # Failure Modes: ValueError on dimension mismatch; RuntimeError on OOM
    # Verification:  tests/kda/test_state_manager.py
    # References:    arXiv:2510.26692 §3
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        num_heads: int,
        max_batch_size: int = 32,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        if key_dim <= 0 or value_dim <= 0:
            raise ValueError(
                f"Dimensions must be positive: key_dim={key_dim}, value_dim={value_dim}"
            )
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive: {num_heads}")
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive: {max_batch_size}")

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.max_batch_size = max_batch_size
        self.dtype = dtype
        self.device = device

        # Pre-allocated state buffer: (max_batch_size, H, K, V)
        self.register_buffer(
            "state_buffer",
            torch.zeros(
                max_batch_size, num_heads, key_dim, value_dim,
                dtype=dtype, device=device,
            ),
        )

        self.current_batch_size: int = 0
        self.memory_allocated: int = 0

        # Checkpointing
        self.enable_checkpointing: bool = False
        self.checkpoint_interval: int = 1000
        self.checkpoints: Dict[int, torch.Tensor] = {}

        # Instrumentation
        self._upd_time_ms: float = 0.0
        self._upd_calls: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-SM-INIT-001
    # Requirement:   Zero-initialise or load an external state into the buffer
    #                for a given batch_size ≤ max_batch_size.
    # Purpose:       Reset state at the start of a new sequence or context window.
    # Inputs:        batch_size ∈ [1, max_batch_size]; initial_state: optional
    #                tensor of shape (batch_size, H, K, V) on any device/dtype.
    # Outputs:       state ∈ R^(batch_size×H×K×V)  (cloned from buffer)
    # Preconditions: batch_size ≤ max_batch_size
    # Postconditions:current_batch_size == batch_size; memory_allocated updated
    # Failure Modes: ValueError if batch_size > max_batch_size or shape mismatch;
    #                RuntimeError wraps any unexpected exception.
    # Verification:  tests/kda/test_state_manager.py::test_initialize_state
    # ─────────────────────────────────────────────────────────────────────────
    def initialize_state(
        self,
        batch_size: int,
        initial_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"batch_size {batch_size} exceeds max_batch_size {self.max_batch_size}."
            )

        try:
            self.current_batch_size = batch_size

            if initial_state is not None:
                expected = (batch_size, self.num_heads, self.key_dim, self.value_dim)
                if initial_state.shape != expected:
                    raise ValueError(
                        f"initial_state shape {initial_state.shape} != expected {expected}"
                    )
                self.state_buffer[:batch_size] = initial_state.to(
                    dtype=self.dtype, device=self.state_buffer.device
                )
            else:
                self.state_buffer[:batch_size].zero_()

            bits = torch.finfo(self.dtype).bits
            self.memory_allocated = (
                batch_size * self.num_heads * self.key_dim * self.value_dim * bits // 8
            )

            return self.state_buffer[:batch_size].clone()

        except ValueError:
            raise
        except Exception as exc:
            logger.error("Failed to initialise state: %s", exc)
            raise RuntimeError(f"Failed to initialise state: {exc}") from exc

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-SM-UPD-001
    # Requirement:   Apply the KDA memory update rule in one call:
    #                S_t = Diag(α_t)·S_{t-1} − β_t·k_t·(k_t^T·Diag(α_t)·S_{t-1})
    #                      + β_t·k_t·v_t^T
    # Purpose:       Fuse diagonal decay, delta-rule correction, and KV write into
    #                a single method so callers need not understand the internals.
    # Rationale:     The three-term form is algebraically equivalent to the DeltaNet
    #                update (Schlag 2021) with head-level gating; see §3.2 in paper.
    #                All contractions are expressed via einsum for clarity and to
    #                enable future Triton/CUDA replacement at this boundary.
    # Inputs:        state ∈ R^(B×H×K×V); keys ∈ R^(B×H×K) (L2-normalised);
    #                values ∈ R^(B×H×V); gates α_t ∈ (0,1)^(B×H×K);
    #                beta β_t ∈ (0,1)^(B×H×1); step: int ≥ 0; return_timing: bool
    # Outputs:       new_state ∈ R^(B×H×K×V); elapsed_ms: float | None
    # Preconditions: keys L2-normalised; gates, beta in (0,1);
    #                state.shape == (B, H, K, V)
    # Postconditions:new_state.shape == state.shape; no NaN in returned tensor
    # Side Effects:  checkpoints dict updated at multiples of checkpoint_interval;
    #                _upd_time_ms and _upd_calls incremented when return_timing
    # Failure Modes: NaN → warned + returns previous state;
    #                Inf → warned + clamped to ±1e6;
    #                OOM → CUDA cache cleared + re-raised
    # Verification:  tests/kda/test_state_manager.py::test_update_state
    # References:    arXiv:2510.26692 Eq. (3); DeltaNet (Schlag et al. 2021)
    # ─────────────────────────────────────────────────────────────────────────
    def update_state(
        self,
        state: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        gates: torch.Tensor,
        beta: torch.Tensor,
        step: int = 0,
        return_timing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        t0 = time.perf_counter() if return_timing else None

        B, H, K, V = state.shape
        if keys.shape != (B, H, K):
            raise ValueError(f"keys shape {keys.shape} != expected {(B, H, K)}")
        if values.shape != (B, H, V):
            raise ValueError(f"values shape {values.shape} != expected {(B, H, V)}")
        if gates.shape != (B, H, K):
            raise ValueError(f"gates shape {gates.shape} != expected {(B, H, K)}")

        try:
            # ── Step 1: diagonal decay ─────────────────────────────────────
            state_decayed = gates.unsqueeze(-1) * state    # (B, H, K, V)

            # ── Step 2: delta-rule correction ─────────────────────────────
            # k_t^T · S_{decayed}  →  (B, H, V)
            kt_S = torch.einsum("bhk,bhkv->bhv", keys, state_decayed)
            # β_t · k_t · (k_t^T · S_{decayed})  →  (B, H, K, V)
            beta_4d = beta.unsqueeze(-1)                   # (B, H, 1, 1)
            correction = beta_4d * torch.einsum("bhk,bhv->bhkv", keys, kt_S)

            # ── Step 3: new KV association ─────────────────────────────────
            kv_write = beta_4d * torch.einsum("bhk,bhv->bhkv", keys, values)

            new_state = state_decayed - correction + kv_write

            # ── Numerical guards ───────────────────────────────────────────
            if torch.isnan(new_state).any():
                warnings.warn(
                    "NaN in KDA state update at step %d; keeping previous state." % step,
                    stacklevel=2,
                )
                new_state = state.clone()
            elif torch.isinf(new_state).any():
                warnings.warn(
                    "Inf in KDA state update at step %d; clamping." % step,
                    stacklevel=2,
                )
                new_state = new_state.clamp(-1e6, 1e6)

            # ── Optional checkpointing ─────────────────────────────────────
            if self.enable_checkpointing and step % self.checkpoint_interval == 0:
                self.checkpoints[step] = new_state.detach().clone()
                if len(self.checkpoints) > 10:
                    del self.checkpoints[min(self.checkpoints)]

            if return_timing:
                elapsed = (time.perf_counter() - t0) * 1_000
                self._upd_time_ms += elapsed
                self._upd_calls += 1
                return new_state, elapsed

            return new_state, None

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.error(
                    "OOM in StateManager.update_state (state %s).", state.shape
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise
        except Exception as exc:
            logger.error("update_state failed: %s", exc)
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-SM-CKPT-001
    # Requirement:   Return the state tensor saved at the given step, or None.
    # Outputs:       tensor ∈ R^(B×H×K×V) | None
    # ─────────────────────────────────────────────────────────────────────────
    def load_checkpoint(self, step: int) -> Optional[torch.Tensor]:
        return self.checkpoints.get(step)

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-SM-CKPT-002
    # Requirement:   Delete all saved checkpoints and free their memory.
    # Side Effects:  self.checkpoints is cleared
    # ─────────────────────────────────────────────────────────────────────────
    def clear_checkpoints(self) -> None:
        self.checkpoints.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-SM-MEM-001
    # Requirement:   Return buffer and checkpoint memory usage in MB.
    # Outputs:       dict with keys 'buffer_mb', 'checkpoints_mb', 'total_mb'
    # ─────────────────────────────────────────────────────────────────────────
    def get_memory_usage(self) -> Dict[str, float]:
        buf_mb = (
            self.state_buffer.numel() * self.state_buffer.element_size() / 1_048_576
        )
        ckpt_mb = sum(
            c.numel() * c.element_size() / 1_048_576
            for c in self.checkpoints.values()
        )
        return {"buffer_mb": buf_mb, "checkpoints_mb": ckpt_mb, "total_mb": buf_mb + ckpt_mb}

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-SM-METRICS-001
    # Requirement:   Return mean update latency in milliseconds.
    # Outputs:       float ≥ 0; 0.0 if no updates recorded
    # ─────────────────────────────────────────────────────────────────────────
    def get_average_update_time(self) -> float:
        if self._upd_calls == 0:
            return 0.0
        return self._upd_time_ms / self._upd_calls

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-SM-METRICS-002
    # Requirement:   Reset timing counters to zero for a fresh profiling window.
    # Side Effects:  Clears _upd_time_ms and _upd_calls
    # ─────────────────────────────────────────────────────────────────────────
    def reset_timing(self) -> None:
        self._upd_time_ms = 0.0
        self._upd_calls = 0

    @property
    def update_time(self) -> float:
        return self._upd_time_ms

    @property
    def update_calls(self) -> int:
        return self._upd_calls

    def extra_repr(self) -> str:
        return (
            f"key_dim={self.key_dim}, value_dim={self.value_dim}, "
            f"num_heads={self.num_heads}, max_batch_size={self.max_batch_size}, "
            f"dtype={self.dtype}"
        )
