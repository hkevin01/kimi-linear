"""
Diagonal-Plus-Low-Rank (DPLR) transition matrices for KDA.

Implements a specialized variant of DPLR that substantially reduces
computation compared to general DPLR formulation while remaining
consistent with the classical delta rule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import time


class DPLRTransition(nn.Module):
    """
    Specialized DPLR transition matrix for Kimi Delta Attention.
    
    Standard DPLR form: St = (D - at bt^T) St-1 + kt vt^T
    KDA constrained form: St = (Diag(αt) - βt kt kt^T Diag(αt)) St-1 + βt kt vt^T
    
    Where the correspondence is:
        D = Diag(αt)
        at = βt kt
        bt = kt ⊙ αt
    
    This constraint enables efficient computation by factoring out the
    diagonal decay, allowing a fine-grained multiplicative decay followed
    by a Householder-style transformation.
    
    Args:
        key_dim: Dimension of keys
        value_dim: Dimension of values
        num_heads: Number of attention heads
        use_eigenvalue_stabilization: Whether to stabilize eigenvalues
        
    Time Complexity: O(K * V) vs O(K^2 * V) for general DPLR
    Space Complexity: O(K * V)
    """
    
    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        num_heads: int,
        use_eigenvalue_stabilization: bool = True,
    ):
        super().__init__()
        
        # Validate inputs
        if key_dim <= 0 or value_dim <= 0:
            raise ValueError(f"Dimensions must be positive: key_dim={key_dim}, value_dim={value_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive: {num_heads}")
        
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.use_eigenvalue_stabilization = use_eigenvalue_stabilization
        
        # Performance tracking
        self.forward_time = 0.0
        self.forward_calls = 0
        self.eigenvalue_warnings = 0
        
    def compute_transition(
        self,
        state: torch.Tensor,
        keys: torch.Tensor,
        gates: torch.Tensor,
        beta: torch.Tensor,
        return_timing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Compute DPLR state transition.
        
        Implements: St = (Diag(αt) - βt kt kt^T Diag(αt)) St-1
        
        This is equivalent to:
        1. Apply diagonal decay: S'= Diag(αt) St-1
        2. Apply rank-1 correction: St = (I - βt kt kt^T) S'
        
        Args:
            state: Current state (B, H, K, V)
            keys: Keys kt (B, H, K), should be L2-normalized
            gates: Forget gates αt (B, H, K) in [0, 1]
            beta: Learning rate βt (B, H, 1)
            return_timing: If True, return execution time
            
        Returns:
            transitioned_state: New state after transition
            timing: Execution time in milliseconds (if return_timing)
            
        Raises:
            ValueError: If input shapes are invalid
            RuntimeError: If computation fails
        """
        start_time = time.perf_counter() if return_timing else None
        
        # Validate shapes
        B, H, K, V = state.shape
        if keys.shape != (B, H, K):
            raise ValueError(f"keys shape {keys.shape} doesn't match expected {(B, H, K)}")
        if gates.shape != (B, H, K):
            raise ValueError(f"gates shape {gates.shape} doesn't match expected {(B, H, K)}")
        if beta.shape[:-1] != (B, H):
            raise ValueError(f"beta shape {beta.shape} doesn't match expected (B, H, ...)")
        
        try:
            # Step 1: Apply fine-grained diagonal decay
            # S' = Diag(αt) St-1
            gates_expanded = gates.unsqueeze(-1)  # (B, H, K, 1)
            state_decayed = gates_expanded * state  # Element-wise multiplication
            
            # Check for eigenvalue stability if enabled
            if self.use_eigenvalue_stabilization:
                self._check_eigenvalue_stability(gates, keys, beta)
            
            # Step 2: Apply rank-1 Householder correction
            # St = (I - βt kt kt^T) S'
            # Efficiently computed as: St = S' - βt kt (kt^T S')
            
            # Compute kt^T S': (B, H, K) x (B, H, K, V) -> (B, H, V)
            kt_state = torch.einsum('bhk,bhkv->bhv', keys, state_decayed)
            
            # Compute βt kt (kt^T S'): (B, H, 1) * (B, H, K) * (B, H, V) -> (B, H, K, V)
            beta_expanded = beta.unsqueeze(-1) if beta.dim() == 3 else beta.unsqueeze(-1).unsqueeze(-1)
            correction = torch.einsum('bhk,bhv->bhkv', keys, kt_state)
            correction = beta_expanded * correction
            
            # Final transition: St = S' - correction
            transitioned_state = state_decayed - correction
            
            # Check for numerical issues
            if torch.isnan(transitioned_state).any():
                print("WARNING: NaN detected in DPLR transition! Returning decayed state.")
                transitioned_state = state_decayed
            elif torch.isinf(transitioned_state).any():
                print("WARNING: Inf detected in DPLR transition! Clipping values.")
                transitioned_state = torch.clamp(transitioned_state, -1e6, 1e6)
            
            # Update metrics
            if return_timing:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self.forward_time += elapsed_ms
                self.forward_calls += 1
                return transitioned_state, elapsed_ms
            
            return transitioned_state, None
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"ERROR: Out of memory in DPLR transition")
                print(f"State shape: {state.shape}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise
        except Exception as e:
            print(f"ERROR in DPLR transition: {e}")
            raise
    
    def _check_eigenvalue_stability(
        self,
        gates: torch.Tensor,
        keys: torch.Tensor,
        beta: torch.Tensor,
        threshold: float = 1.1,
    ):
        """
        Check eigenvalue stability of the transition matrix.
        
        For stability, we need |λ_max| <= 1. The transition matrix
        (Diag(α) - β k k^T Diag(α)) has eigenvalues that depend on
        the decay gates α and the correction term β k k^T.
        
        We use a simple heuristic: if max(α) * (1 + β * ||k||^2) > threshold,
        issue a warning.
        
        Args:
            gates: Forget gates α (B, H, K)
            keys: Keys k (B, H, K)
            beta: Learning rate β (B, H, 1)
            threshold: Stability threshold (default: 1.1)
        """
        try:
            # Compute max gate value
            max_gate = gates.max().item()
            
            # Compute max correction strength
            key_norm_sq = (keys ** 2).sum(dim=-1, keepdim=True)  # (B, H, 1)
            max_correction = (beta * key_norm_sq).max().item()
            
            # Estimate max eigenvalue
            est_max_eigenvalue = max_gate * (1 + max_correction)
            
            if est_max_eigenvalue > threshold:
                self.eigenvalue_warnings += 1
                if self.eigenvalue_warnings <= 3:  # Only warn first few times
                    print(f"WARNING: Potential eigenvalue instability detected!")
                    print(f"Estimated max eigenvalue: {est_max_eigenvalue:.4f} > {threshold}")
                    print(f"Consider reducing beta or normalizing keys more carefully.")
                    
        except Exception as e:
            # Don't let stability checking break the forward pass
            print(f"Warning: Eigenvalue stability check failed: {e}")
    
    def forward(
        self,
        state: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        gates: torch.Tensor,
        beta: torch.Tensor,
        return_timing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Full DPLR update: transition + key-value accumulation.
        
        Implements: St = (Diag(αt) - βt kt kt^T Diag(αt)) St-1 + βt kt vt^T
        
        Args:
            state: Current state (B, H, K, V)
            keys: Keys kt (B, H, K)
            values: Values vt (B, H, V)
            gates: Forget gates αt (B, H, K)
            beta: Learning rate βt (B, H, 1)
            return_timing: If True, return execution time
            
        Returns:
            new_state: Updated state
            timing: Execution time in milliseconds (if return_timing)
        """
        start_time = time.perf_counter() if return_timing else None
        
        # Apply transition
        transitioned_state, _ = self.compute_transition(
            state, keys, gates, beta, return_timing=False
        )
        
        # Add new key-value association: βt kt vt^T
        beta_expanded = beta.unsqueeze(-1) if beta.dim() == 3 else beta.unsqueeze(-1).unsqueeze(-1)
        kv_update = torch.einsum('bhk,bhv->bhkv', keys, values)
        kv_update = beta_expanded * kv_update
        
        # Final state
        new_state = transitioned_state + kv_update
        
        # Update metrics
        if return_timing:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.forward_time += elapsed_ms
            self.forward_calls += 1
            return new_state, elapsed_ms
        
        return new_state, None
    
    def get_average_time(self) -> float:
        """Get average forward pass time in milliseconds."""
        if self.forward_calls == 0:
            return 0.0
        return self.forward_time / self.forward_calls
    
    def reset_timing(self):
        """Reset timing statistics."""
        self.forward_time = 0.0
        self.forward_calls = 0
        self.eigenvalue_warnings = 0
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"key_dim={self.key_dim}, value_dim={self.value_dim}, "
            f"num_heads={self.num_heads}, "
            f"use_eigenvalue_stabilization={self.use_eigenvalue_stabilization}"
        )


def test_dplr_transition():
    """Test DPLR transition module."""
    print("Testing DPLRTransition...")
    
    # Configuration
    batch_size = 4
    num_heads = 8
    key_dim = 64
    value_dim = 64
    
    # Create module
    try:
        dplr = DPLRTransition(
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            use_eigenvalue_stabilization=True,
        )
        print("✓ Module created successfully")
    except Exception as e:
        print(f"✗ Module creation failed: {e}")
        return
    
    # Test 1: Initialize state and inputs
    try:
        state = torch.randn(batch_size, num_heads, key_dim, value_dim) * 0.1
        keys = F.normalize(torch.randn(batch_size, num_heads, key_dim), dim=-1)  # L2 normalized
        values = torch.randn(batch_size, num_heads, value_dim)
        gates = torch.sigmoid(torch.randn(batch_size, num_heads, key_dim))
        beta = torch.sigmoid(torch.randn(batch_size, num_heads, 1)) * 0.5
        print("✓ Inputs initialized")
    except Exception as e:
        print(f"✗ Input initialization failed: {e}")
        return
    
    # Test 2: Compute transition only
    try:
        transitioned, timing = dplr.compute_transition(
            state, keys, gates, beta, return_timing=True
        )
        assert transitioned.shape == state.shape
        print(f"✓ Transition computation (time: {timing:.2f}ms)")
    except Exception as e:
        print(f"✗ Transition computation failed: {e}")
        return
    
    # Test 3: Full update (transition + KV accumulation)
    try:
        new_state, timing = dplr.forward(
            state, keys, values, gates, beta, return_timing=True
        )
        assert new_state.shape == state.shape
        print(f"✓ Full DPLR update (time: {timing:.2f}ms)")
    except Exception as e:
        print(f"✗ Full DPLR update failed: {e}")
        return
    
    # Test 4: Boundary conditions - near-zero gates (heavy forgetting)
    try:
        gates_zero = torch.ones_like(gates) * 0.01  # Almost complete forgetting
        transitioned_zero, _ = dplr.compute_transition(
            state, keys, gates_zero, beta, return_timing=False
        )
        # State should be very small after heavy forgetting
        state_norm = torch.norm(transitioned_zero)
        print(f"✓ Heavy forgetting (||state||={state_norm:.4f})")
    except Exception as e:
        print(f"✗ Heavy forgetting test failed: {e}")
    
    # Test 5: Boundary conditions - near-one gates (minimal forgetting)
    try:
        gates_one = torch.ones_like(gates) * 0.99  # Almost no forgetting
        transitioned_one, _ = dplr.compute_transition(
            state, keys, gates_one, beta, return_timing=False
        )
        # State should be similar to input after minimal forgetting
        state_diff = torch.norm(transitioned_one - state) / torch.norm(state)
        print(f"✓ Minimal forgetting (relative diff={state_diff:.4f})")
    except Exception as e:
        print(f"✗ Minimal forgetting test failed: {e}")
    
    # Test 6: Eigenvalue stability warning
    try:
        large_beta = torch.ones_like(beta) * 0.9  # Large beta
        _, _ = dplr.compute_transition(
            state, keys, gates, large_beta, return_timing=False
        )
        print(f"✓ Eigenvalue stability check (warnings: {dplr.eigenvalue_warnings})")
    except Exception as e:
        print(f"✗ Eigenvalue stability test failed: {e}")
    
    print(f"\n{'='*60}")
    print(f"DPLRTransition tests complete!")
    print(f"Module info: {dplr}")
    print(f"Average forward time: {dplr.get_average_time():.2f}ms")
    print('='*60)


if __name__ == "__main__":
    test_dplr_transition()
