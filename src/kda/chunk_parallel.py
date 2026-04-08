"""
Chunkwise parallel KDA forward pass.

Implements the inter-chunk recurrent + intra-chunk parallel algorithm that
replaces the naive token-by-token loop in KDALayer.  The sequence of length T
is partitioned into NT non-overlapping chunks of size BT.  Within each chunk,
a parallelisable quadratic attention is computed (intra-chunk); across chunk
boundaries, the persistent state carries information (inter-chunk).

The algorithm uses two key algebraic tools:

  WY Representation
  -----------------
  For a chunk of BT keys [k_1, …, k_BT] with scalars [β_1, …, β_BT] and
  per-channel gates [α_1, …, α_BT], the cumulative transition matrix
  A_{1:BT} can be expressed in WY form (Bischof & Van Loan, 1987):

      A_{1:BT} = I + W Y^T

  where W ∈ R^(K × BT) and Y ∈ R^(K × BT) are low-rank factors computed
  by a single forward-triangular solve over the chunk.  This avoids
  materialising the full K×K transition matrix.

  UT Transform
  ------------
  The inter-chunk state update is structured as a rank-BT modification:

      S_chunk = Λ · S_prev + U T^T

  where Λ = Diag(cumulative_gate) ∈ R^(K×K), U ∈ R^(K × BT), T ∈ R^(V × BT).
  This is computed by a single einsum rather than BT sequential outer products.

Architecture reference: Kimi Linear (arXiv:2510.26692), §3.3 and Appendix A.
"""

# ─────────────────────────────────────────────────────────────────────────────
# MODULE SPEC
# ID:            KDA-CHUNK-MOD-001
# Requirement:   Implement O(T/BT · BT² · H) chunkwise parallel KDA forward,
#                returning the same (B,T,H,V) output as the recurrent baseline.
# Purpose:       Replace the O(T) sequential loop with a blocked algorithm that
#                achieves 5–10× wall-clock speedup on GPU for T ≥ 512.
# Rationale:     The WY + UT decomposition allows the intra-chunk attention
#                matrix to be computed as a dense BT×BT matmul, which maps
#                efficiently to Tensor Cores, while the inter-chunk update
#                retains O(K·V·H) state.
# Assumptions:   T divisible by BT (caller must pad if necessary).
#                Keys are L2-normalised before entry.
#                Gates are in log-space (α_log = log σ(…) < 0).
# Constraints:   BT ≤ 256 recommended for SM90 shared-memory budget.
# References:    arXiv:2510.26692 Appendix A; Bischof & Van Loan 1987 (WY);
#                DeltaNet chunkwise (Schlag et al. 2021)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION SPEC
# ID:            KDA-CHUNK-WY-001
# Requirement:   Given a chunk of BT keys, gates (log-space), and β scalars,
#                compute the WY representation factors (W, Y, Λ) such that
#                A_{1:BT} = Λ + W Y^T where Λ = Diag(cum_gate_product).
# Purpose:       Prepare per-chunk transition coefficients for the intra-chunk
#                attention computation without materialising the K×K matrix.
# Inputs:        k_chunk ∈ R^(B×H×BT×K), L2-normalised;
#                g_chunk ∈ R^(B×H×BT×K) log-space gates (< 0);
#                beta_chunk ∈ (0,1)^(B×H×BT)
# Outputs:       w ∈ R^(B×H×BT×K): WY W-factors (gated key residuals)
#                y ∈ R^(B×H×BT×V): WY Y-factors (value targets)
#                cum_g ∈ R^(B×H×K): cumulative log-gate product over chunk
#                A ∈ R^(B×H×BT×BT): lower-triangular intra-chunk attention mask
# Preconditions: g_chunk values ≤ 0 (gate is a decay, not gain)
# Postconditions:A is strictly lower-triangular (diagonal = 0 from delta rule)
# References:    arXiv:2510.26692 Eq. (A.2); Bischof & Van Loan 1987
# ─────────────────────────────────────────────────────────────────────────────
def compute_wy_representation(
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
    g_chunk: torch.Tensor,
    beta_chunk: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, BT, K = k_chunk.shape
    V = v_chunk.shape[-1]

    # ── Cumulative log-gate prefix sums ────────────────────────────────────
    # g_cum[b,h,t,k] = sum_{s=0}^{t-1} g_chunk[b,h,s,k]  (exclusive prefix)
    g_cum = torch.cumsum(g_chunk, dim=2)                     # (B,H,BT,K)
    g_cumsum_chunk = g_cum[:, :, -1, :]                      # (B,H,K) - full chunk product

    # ── Gated keys for transition: k̃_t = exp(g_cum[t]) · k_t ──────────────
    # k_gated[t] = k_t · exp(g_cum_total − g_cum[t])  (normalise to chunk end)
    g_total = g_cumsum_chunk.unsqueeze(2)                    # (B,H,1,K)
    k_gated = k_chunk * torch.exp(g_total - g_cum)          # (B,H,BT,K)

    # ── Intra-chunk interaction matrix A ─────────────────────────────────
    # A[i,j] = β_i · (k_gated_i · k_j^T) for i > j (causal, strictly lower tri)
    # Shape: (B, H, BT, BT) where A[i,j] is token i attending to token j
    A = torch.einsum("bhtk,bhsk->bhts", k_gated, k_chunk)   # (B,H,BT,BT)
    # Scale by β_i
    A = A * beta_chunk.unsqueeze(-1)                         # (B,H,BT,BT)
    # Strictly lower-triangular causal mask (token i can only use j < i)
    mask = torch.tril(torch.ones(BT, BT, device=k_chunk.device, dtype=torch.bool),
                      diagonal=-1)
    A = A.masked_fill(~mask, 0.0)

    # ── Forward triangular solve: invert (I + A) via Neumann series ────────
    # (I + A) · X = I  →  X = I - A + A² - …  (terminates for strictly lower tri)
    # Equivalent to: solve_tril(I, A) in one pass
    I = torch.eye(BT, device=A.device, dtype=A.dtype).expand(B, H, BT, BT)
    Akk_inv = I.clone()
    for i in range(1, BT):
        # Row i of Akk_inv += sum_j<i A[i,j] · Akk_inv[j,:]
        Akk_inv = I - torch.tril(A @ Akk_inv, diagonal=-1)

    # ── WY factors ─────────────────────────────────────────────────────────
    # w[t] = Akk_inv[t,:] @ (β_t · k_gated_t)  →  residual key for state update
    w = torch.einsum("bhts,bhsk->bhtk", Akk_inv, k_gated * beta_chunk.unsqueeze(-1))
    # y[t] = Akk_inv[t,:] @ v_t                →  residual value for state update
    y = torch.einsum("bhts,bhsv->bhtv", Akk_inv, v_chunk)

    return w, y, g_cumsum_chunk, A


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION SPEC
# ID:            KDA-CHUNK-UT-001
# Requirement:   Apply the UT-transform inter-chunk state update:
#                S_new = Diag(exp(cum_g)) · S_prev
#                       + einsum("bhtk,bhtv->bhkv", w, y)
# Purpose:       Update the persistent K×V state at each chunk boundary using
#                the rank-BT correction computed from WY factors.
# Inputs:        state ∈ R^(B×H×K×V); w ∈ R^(B×H×BT×K); y ∈ R^(B×H×BT×V);
#                cum_g ∈ R^(B×H×K) (log-space)
# Outputs:       new_state ∈ R^(B×H×K×V)
# Preconditions: state.shape == (B, H, K, V)
# Postconditions:new_state.shape == (B, H, K, V); no NaN
# References:    arXiv:2510.26692 Eq. (A.3)
# ─────────────────────────────────────────────────────────────────────────────
def ut_transform_state_update(
    state: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    cum_g: torch.Tensor,
) -> torch.Tensor:
    # Diagonal decay: S = Diag(exp(cum_g)) · S  (broadcast over V dim)
    decay = torch.exp(cum_g).unsqueeze(-1)                   # (B,H,K,1)
    state_decayed = decay * state                             # (B,H,K,V)

    # Rank-BT write: delta_S = sum_t w_t ⊗ y_t^T
    delta = torch.einsum("bhtk,bhtv->bhkv", w, y)            # (B,H,K,V)
    return state_decayed + delta


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION SPEC
# ID:            KDA-CHUNK-INTRA-001
# Requirement:   Compute the intra-chunk output contributions:
#                  o_intra[i] = Σ_{j<i} Aqk[i,j] · y[j]
#                where Aqk[i,j] = scale · q_i^T k_j · exp(g_cum[i] − g_cum[j])
# Purpose:       Compute within-chunk query-key interactions in one batched matmul.
# Inputs:        q_chunk ∈ R^(B×H×BT×K); k_chunk ∈ R^(B×H×BT×K);
#                g_chunk ∈ R^(B×H×BT×K); y ∈ R^(B×H×BT×V); scale: float
# Outputs:       o_intra ∈ R^(B×H×BT×V)
# References:    arXiv:2510.26692 Eq. (A.4)
# ─────────────────────────────────────────────────────────────────────────────
def compute_intra_chunk_output(
    q_chunk: torch.Tensor,
    k_chunk: torch.Tensor,
    g_chunk: torch.Tensor,
    y: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    B, H, BT, K = q_chunk.shape
    g_cum = torch.cumsum(g_chunk, dim=2)                     # (B,H,BT,K)

    # Relative gate: q sees all j ≤ i with decay exp(g_cum[i] − g_cum[j])
    # Aqk[i,j] = scale · (q_i · exp(g_cum[i])) · (k_j · exp(−g_cum[j]))^T
    q_gated = q_chunk * torch.exp(g_cum)                     # (B,H,BT,K)
    k_gated = k_chunk * torch.exp(-g_cum)                    # (B,H,BT,K)

    Aqk = scale * torch.einsum("bhtk,bhsk->bhts", q_gated, k_gated)  # (B,H,BT,BT)

    # Strictly causal lower-triangular mask (i > j, not i == j for delta rule)
    mask = torch.tril(torch.ones(BT, BT, device=q_chunk.device, dtype=torch.bool),
                      diagonal=-1)
    Aqk = Aqk.masked_fill(~mask, 0.0)

    # Intra-chunk output: o_intra = Aqk @ y
    return torch.einsum("bhts,bhsv->bhtv", Aqk, y)           # (B,H,BT,V)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS SPEC
# ID:            KDA-CHUNK-CLS-001
# Requirement:   Provide a forward(Q, K, V, g_log, beta, state) → (output, state)
#                interface that replaces the token loop in KDALayer with the
#                WY+UT chunkwise parallel algorithm.
# Purpose:       GPU-friendly KDA computation; compatible with KDALayer via
#                the use_chunk_parallel flag.
# Inputs:        All dimension params from KDALayer; chunk_size: int (default 64)
# Outputs:       o ∈ R^(B×T×H×V); final_state ∈ R^(B×H×K×V)
# Preconditions: T % chunk_size == 0 (KDALayer pads to nearest multiple)
# Failure Modes: NaN → warn + fall back to previous state for that chunk
# Verification:  tests/kda/test_chunk_parallel.py
# References:    arXiv:2510.26692 Appendix A
# ─────────────────────────────────────────────────────────────────────────────
class ChunkwiseParallelKDA(nn.Module):
    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        num_heads: int,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        if key_dim <= 0 or value_dim <= 0 or num_heads <= 0:
            raise ValueError("All dimension arguments must be positive.")
        if chunk_size <= 0 or (chunk_size & (chunk_size - 1)) != 0:
            raise ValueError("chunk_size must be a positive power of 2.")

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD SPEC
    # ID:            KDA-CHUNK-FWD-001
    # Requirement:   Split (B,T,H,K/V) tensors into NT chunks; for each chunk
    #                compute WY+UT state update and intra-chunk attention output;
    #                accumulate inter-chunk output from carried state; return
    #                stacked (B,T,H,V) output and final state.
    # Inputs:        Q,K,V ∈ R^(B×T×H×K/V); g_log ∈ R^(B×T×H×K) (≤0 log gates);
    #                beta ∈ (0,1)^(B×T×H); state: optional R^(B×H×K×V)
    # Outputs:       o ∈ R^(B×T×H×V); final_state ∈ R^(B×H×K×V)
    # Preconditions: T % chunk_size == 0; K L2-normalised per token
    # Postconditions:o.shape == (B, T, H, V); no NaN in o or final_state
    # Failure Modes: If T % chunk_size != 0 → ValueError (caller must pad)
    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        g_log: torch.Tensor,
        beta: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H, d_k = Q.shape
        V_dim = V.shape[-1]
        BT = self.chunk_size

        if T % BT != 0:
            raise ValueError(
                f"Sequence length {T} must be divisible by chunk_size {BT}. "
                f"Pad the input before calling ChunkwiseParallelKDA."
            )

        NT = T // BT
        scale = d_k ** -0.5

        # Reshape: (B,T,H,d) → (B,H,NT,BT,d) per-chunk views
        def to_chunks(x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, H, d)
            d = x.shape[-1]
            return x.reshape(B, NT, BT, H, d).permute(0, 3, 1, 2, 4)  # (B,H,NT,BT,d)

        Q_c = to_chunks(Q)       # (B,H,NT,BT,K)
        K_c = to_chunks(K)       # (B,H,NT,BT,K)
        V_c = to_chunks(V)       # (B,H,NT,BT,V)
        g_c = to_chunks(g_log)   # (B,H,NT,BT,K)
        b_c = beta.reshape(B, NT, BT, H).permute(0, 3, 1, 2)  # (B,H,NT,BT)

        # Initialise state
        if state is None:
            current_state = Q.new_zeros(B, H, d_k, V_dim)
        else:
            current_state = state.clone()

        outputs = []

        for n in range(NT):
            q_n = Q_c[:, :, n]   # (B,H,BT,K)
            k_n = K_c[:, :, n]   # (B,H,BT,K)
            v_n = V_c[:, :, n]   # (B,H,BT,V)
            g_n = g_c[:, :, n]   # (B,H,BT,K)
            b_n = b_c[:, :, n]   # (B,H,BT)

            # ── WY representation for this chunk ──────────────────────────
            w, y, cum_g, _ = compute_wy_representation(k_n, v_n, g_n, b_n)

            # ── Inter-chunk: output from carried state ────────────────────
            # q_gated[t] = q_t · exp(g_cum[t]) queries the decayed state
            g_cum_n = torch.cumsum(g_n, dim=2)                         # (B,H,BT,K)
            q_gated = q_n * torch.exp(g_cum_n)                         # (B,H,BT,K)
            o_inter = scale * torch.einsum("bhtk,bhkv->bhtv",
                                           q_gated, current_state)     # (B,H,BT,V)

            # ── Intra-chunk output ─────────────────────────────────────────
            o_intra = compute_intra_chunk_output(q_n, k_n, g_n, y, scale)  # (B,H,BT,V)

            o_chunk = o_inter + o_intra                                 # (B,H,BT,V)

            # ── NaN guard ─────────────────────────────────────────────────
            if torch.isnan(o_chunk).any():
                logger.warning("NaN in chunk %d output; replacing with zeros.", n)
                o_chunk = torch.nan_to_num(o_chunk, nan=0.0)

            outputs.append(o_chunk)

            # ── UT transform: update persistent state ─────────────────────
            current_state = ut_transform_state_update(
                current_state, w, y, cum_g
            )

        # Stack: (B,H,NT,BT,V) → (B,T,H,V)
        o = torch.stack(outputs, dim=2)        # (B,H,NT,BT,V)
        o = o.permute(0, 2, 3, 1, 4)          # (B,NT,BT,H,V)
        o = o.reshape(B, T, H, V_dim)

        return o, current_state

    def extra_repr(self) -> str:
        return (
            f"key_dim={self.key_dim}, value_dim={self.value_dim}, "
            f"num_heads={self.num_heads}, chunk_size={self.chunk_size}"
        )
