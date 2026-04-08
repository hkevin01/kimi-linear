"""
KDA Layer: assembled Kimi Delta Attention recurrent module.

Combines FineGrainedGating, DPLRTransition, and StateManager into a single
nn.Module that maps (B, T, D) → (B, T, D) with O(1) state per position,
plus architectural enhancements taken from arXiv:2510.26692 §3.1–3.2:

  • Depthwise short convolution (kernel=4) on the key projection (§3.1).
  • Per-head RMSNorm on retrieved content before the output gate (§3.2).
  • Low-rank sigmoid output gate (§3.2): σ(W_up W_down x) ⊙ norm(retrieved).
"""

# ─────────────────────────────────────────────────────────────────────────────
# MODULE SPEC
# ID:            KDA-LAYER-MOD-001
# Requirement:   Implement the complete KDA recurrent layer including projections,
#                gating, DPLR update, and output gate as described in §3 of the Kimi
#                Linear paper.
# Purpose:       Provide a drop-in nn.Module for KDA layers in hybrid LLM stacks
#                (3:1 KDA-to-MLA ratio per §4 deployment configuration).
# Rationale:     Assembling all components behind a single forward() interface
#                makes it easy to stack multiple KDA layers, share state across
#                chunks, and replace individual sub-components with CUDA kernels.
# Assumptions:   Input is already layer-normed by the calling transformer block.
#                Sequence dimension is the second axis (B, T, D).
# References:    arXiv:2510.26692 §3.1, §3.2, §3.3
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gating import FineGrainedGating
from .dplr import DPLRTransition
from .state_manager import StateManager
from .chunk_parallel import ChunkwiseParallelKDA

logger = logging.getLogger(__name__)


class KDALayer(nn.Module):
    # ─────────────────────────────────────────────────────────────────────────
    # CLASS SPEC
    # ID:            KDA-LAYER-CLS-001
    # Requirement:   Given (B, T, D) input, produce (B, T, D) output by applying
    #                a recurrent KDA cell at each token position with:
    #                (1) QKV projections, (2) short conv on K, (3) forget gating,
    #                (4) DPLR state update, (5) content retrieval, (6) RMSNorm,
    #                (7) output gate, (8) output projection.
    # Purpose:       Single-call interface for KDA computation; compatible with
    #                standard transformer block APIs.
    # Inputs:        hidden_dim ∈ Z+; num_heads ∈ Z+; head_dim ∈ Z+;
    #                dropout ∈ [0,1); max_batch_size ∈ Z+;
    #                use_short_conv: bool; use_output_gate: bool;
    #                output_gate_rank: int ≥ 1.
    # Failure Modes: OOM → logs error, clears CUDA cache, re-raises;
    #                NaN state → reported via StateManager guard
    # Verification:  tests/kda/test_kda_layer.py, tests/kda/test_integration.py
    # References:    arXiv:2510.26692 §3.1–3.3
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        max_batch_size: int = 32,
        use_short_conv: bool = True,
        use_output_gate: bool = True,
        output_gate_rank: int = 0,  # 0 → hidden_dim // 4
        use_chunk_parallel: bool = False,
        chunk_size: int = 64,
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
        self.use_short_conv = use_short_conv
        self.use_output_gate = use_output_gate
        self.use_chunk_parallel = use_chunk_parallel
        self.chunk_size = chunk_size

        # ── Projections ────────────────────────────────────────────────────
        self.q_proj = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, hidden_dim, bias=False)

        # ── Short convolution on K (§3.1) ──────────────────────────────────
        # Depthwise conv1d: kernel=4, causal padding via .narrow() after conv
        if use_short_conv:
            self.k_conv = nn.Conv1d(
                self.inner_dim, self.inner_dim,
                kernel_size=4, padding=3,
                groups=self.inner_dim, bias=False,
            )

        # ── β scalar per head (DeltaNet-style) ────────────────────────────
        self.beta_proj = nn.Linear(hidden_dim, num_heads, bias=True)

        # ── Fine-grained channel gating ────────────────────────────────────
        self.gating = FineGrainedGating(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        # ── DPLR transition ────────────────────────────────────────────────
        self.dplr = DPLRTransition(
            key_dim=head_dim,
            value_dim=head_dim,
            num_heads=num_heads,
        )

        # ── State manager ──────────────────────────────────────────────────
        self.state_manager = StateManager(
            key_dim=head_dim,
            value_dim=head_dim,
            num_heads=num_heads,
            max_batch_size=max_batch_size,
        )

        # ── Chunkwise parallel engine (optional) ───────────────────────────
        if use_chunk_parallel:
            self.chunk_engine: Optional[ChunkwiseParallelKDA] = ChunkwiseParallelKDA(
                key_dim=head_dim,
                value_dim=head_dim,
                num_heads=num_heads,
                chunk_size=chunk_size,
            )
        else:
            self.chunk_engine = None

        # ── Output normalisation (§3.2) ────────────────────────────────────
        self.out_norm = nn.RMSNorm(head_dim)

        # ── Low-rank output gate (§3.2) ────────────────────────────────────
        if use_output_gate:
            rank = output_gate_rank if output_gate_rank > 0 else max(1, hidden_dim // 4)
            self.gate_proj_down = nn.Linear(hidden_dim, rank, bias=False)
            self.gate_proj_up = nn.Linear(rank, self.inner_dim, bias=False)

        # Instrumentation
        self._fwd_time_ms: float = 0.0
        self._fwd_calls: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-LAYER-FWD-001
    # Requirement:   For each token t ∈ [0, T):
    #                  q_t = W_Q x_t                          [query]
    #                  k_t = ShortConv(W_K x_t)              [key + short context]
    #                  v_t = W_V x_t                          [value]
    #                  k_t = L2-normalise(k_t)
    #                  α_t = FineGrainedGating(x_t)           [forget gate]
    #                  β_t = σ(W_β x_t)                       [write rate]
    #                  S_t = DPLR(S_{t-1}, k_t, v_t, α_t, β_t)
    #                  o_t = S_t^T q_t                        [retrieve]
    #                  o_t = RMSNorm(o_t)                     [stabilise]
    #                  o_t = o_t ⊙ σ(W_up W_down x_t)        [output gate]
    #                  y_t = W_O o_t                          [project out]
    # Purpose:       Produce contextualised output via constant-memory recurrence.
    # Inputs:        x ∈ R^(B×T×D); state: optional R^(B×H×K×V);
    #                return_state: bool; return_timing: bool
    # Outputs:       output ∈ R^(B×T×D);
    #                final_state: optional R^(B×H×K×V);
    #                elapsed_ms: optional float
    # Preconditions: x is already layer-normed; B ≤ max_batch_size in StateManager
    # Postconditions:output.shape == x.shape; no NaN in output
    # Side Effects:  state_manager internal buffer written; _fwd_* counters updated
    # Failure Modes: OOM → logs error, clears CUDA cache, re-raises
    # Constraints:   O(T·B·H·K·V) time; O(B·H·K·V) additional space per call
    # Verification:  tests/kda/test_kda_layer.py::test_forward_shape,
    #                tests/kda/test_integration.py::test_chunked_sequence
    # References:    arXiv:2510.26692 §3.1 (short conv), §3.2 (output gate), §3.3 (DPLR)
    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        return_state: bool = True,
        return_timing: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        t0 = time.perf_counter() if return_timing else None
        B, T, D = x.shape

        try:
            # ── QKV projections ────────────────────────────────────────────
            Q = self.q_proj(x)                          # (B, T, H*K)
            K = self.k_proj(x)                          # (B, T, H*K)
            V = self.v_proj(x)                          # (B, T, H*V)

            # ── Short convolution on K (causal, no future leakage) ─────────
            if self.use_short_conv:
                # Conv1d expects (B, C, T); trim causal padding
                K = self.k_conv(K.transpose(1, 2)).narrow(2, 0, T).transpose(1, 2)
                K = F.silu(K)

            # ── Reshape to (B, T, H, head_dim) then permute ───────────────
            Q = Q.view(B, T, self.num_heads, self.head_dim)
            K = K.view(B, T, self.num_heads, self.head_dim)
            V = V.view(B, T, self.num_heads, self.head_dim)

            # ── L2-normalise keys ──────────────────────────────────────────
            K = F.normalize(K, p=2, dim=-1)

            # ── β per head ─────────────────────────────────────────────────
            beta_raw = self.beta_proj(x)                # (B, T, H)
            beta = torch.sigmoid(beta_raw)              # (B, T, H) ∈ (0,1)

            # ── Forget gates ───────────────────────────────────────────────
            gates, _ = self.gating(x)                   # (B, T, H, K)

            # ── Initialise or pass-in state ────────────────────────────────
            if state is None:
                current_state = self.state_manager.initialize_state(B)
            else:
                current_state = state.clone()

            # ── Choose computation path ────────────────────────────────────
            if self.use_chunk_parallel and self.chunk_engine is not None:
                # ── Chunkwise parallel path ────────────────────────────────
                # Pad T to multiple of chunk_size if necessary
                pad = (-T) % self.chunk_size
                if pad > 0:
                    Q_in = F.pad(Q, (0, 0, 0, 0, 0, pad))
                    K_in = F.pad(K, (0, 0, 0, 0, 0, pad))
                    V_in = F.pad(V, (0, 0, 0, 0, 0, pad))
                    gates_in = F.pad(gates, (0, 0, 0, 0, 0, pad))
                    beta_in = F.pad(beta, (0, 0, 0, pad))
                else:
                    Q_in, K_in, V_in, gates_in, beta_in = Q, K, V, gates, beta

                # Convert gates from linear-space to log-space for chunk engine
                g_log = torch.log(gates_in.clamp(min=1e-8))  # (B, T_pad, H, K)

                out_hd, current_state = self.chunk_engine(
                    Q_in, K_in, V_in, g_log, beta_in, state=current_state
                )  # out_hd: (B, T_pad, H, V)

                # Trim padding and reshape to (B, T, inner_dim)
                out_hd = out_hd[:, :T, :, :]               # (B, T, H, V)
                out = out_hd.reshape(B, T, self.inner_dim)

            else:
                # ── Recurrent loop over T ──────────────────────────────────
                outputs = []
                for t in range(T):
                    qt = Q[:, t, :, :]              # (B, H, K)
                    kt = K[:, t, :, :]              # (B, H, K)
                    vt = V[:, t, :, :]              # (B, H, V)
                    alpha_t = gates[:, t, :, :]     # (B, H, K)
                    beta_t = beta[:, t, :].unsqueeze(-1)  # (B, H, 1)

                    # DPLR state update: S_t = A_t S_{t-1} + β_t k_t v_t^T
                    new_state, _ = self.dplr.forward(
                        current_state, kt, vt, alpha_t, beta_t
                    )
                    current_state = new_state

                    # Retrieve: o_t = S_t^T q_t  →  einsum (B,H,K,V) × (B,H,K) → (B,H,V)
                    ot = torch.einsum("bhkv,bhk->bhv", current_state, qt)  # (B, H, V)

                    # RMSNorm per head
                    ot = self.out_norm(ot)

                    outputs.append(ot)

                # ── Stack all positions ────────────────────────────────────
                out = torch.stack(outputs, dim=1)   # (B, T, H, V)
                out = out.reshape(B, T, self.inner_dim)

            # ── Output gate: σ(W_up W_down x) ⊙ out ──────────────────────
            if self.use_output_gate:
                gate = torch.sigmoid(self.gate_proj_up(self.gate_proj_down(x)))
                out = out * gate                # element-wise, both (B, T, inner_dim)

            # ── Output projection ──────────────────────────────────────────
            output = self.out_proj(out)         # (B, T, D)

            result: list = [output]
            if return_state:
                result.append(current_state)
            if return_timing:
                elapsed = (time.perf_counter() - t0) * 1_000
                self._fwd_time_ms += elapsed
                self._fwd_calls += 1
                result.append(elapsed)

            return tuple(result) if len(result) > 1 else output

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.error("OOM in KDALayer.forward (x %s).", x.shape)
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
    def forward_calls(self) -> int:
        return self._fwd_calls

    @property
    def forward_time(self) -> float:
        return self._fwd_time_ms

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, use_short_conv={self.use_short_conv}, "
            f"use_output_gate={self.use_output_gate}"
        )
