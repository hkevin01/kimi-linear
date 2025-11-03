"""
Fine-grained gating mechanism for Kimi Delta Attention.

Implements channel-wise decay gates that enable more precise control
over the finite-state RNN memory compared to head-wise gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import time


class FineGrainedGating(nn.Module):
    """
    Fine-grained channel-wise gating mechanism.
    
    Unlike coarse head-wise gates (e.g., Mamba2, GDN), this implementation
    provides independent forgetting rates for each feature dimension,
    enabling more effective use of limited finite-state RNN memory.
    
    Args:
        hidden_dim: Hidden dimension size
        head_dim: Dimension per attention head
        num_heads: Number of attention heads
        rank: Rank for low-rank projection (default: head_dim)
        dropout: Dropout probability (default: 0.0)
        
    Time Complexity: O(d * rank) for forward pass
    Space Complexity: O(d * rank) for parameters
    """
    
    def __init__(
        self,
        hidden_dim: int,
        head_dim: int,
        num_heads: int,
        rank: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Validate inputs with proper error messages
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
            
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.rank = rank if rank is not None else head_dim
        self.dropout = dropout
        
        # Low-rank projection for gate computation
        # This reduces parameters while maintaining expressiveness
        self.gate_down = nn.Linear(hidden_dim, self.rank, bias=False)
        self.gate_up = nn.Linear(self.rank, head_dim * num_heads, bias=False)
        
        # Dropout for regularization
        if self.dropout > 0.0:
            self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights for stable training
        self._initialize_weights()
        
        # Performance tracking
        self.forward_time = 0.0
        self.forward_calls = 0
        
    def _initialize_weights(self):
        """
        Initialize weights for stable training.
        
        Uses Xavier uniform initialization for the down projection
        and small random values for the up projection to start with
        near-identity gates.
        """
        try:
            nn.init.xavier_uniform_(self.gate_down.weight)
            nn.init.uniform_(self.gate_up.weight, -0.01, 0.01)
        except Exception as e:
            # Graceful fallback if initialization fails
            print(f"Warning: Weight initialization failed: {e}")
            print("Using default PyTorch initialization")
    
    def forward(
        self,
        x: torch.Tensor,
        return_timing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Compute channel-wise decay gates.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            return_timing: If True, return execution time
            
        Returns:
            gates: Decay gates in [0, 1] of shape (batch, seq_len, num_heads, head_dim)
            timing: Execution time in milliseconds (if return_timing=True)
            
        Raises:
            ValueError: If input shape is invalid
            RuntimeError: If computation fails (e.g., OOM)
        """
        start_time = time.perf_counter() if return_timing else None
        
        # Validate input shape
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq_len, hidden_dim), got shape {x.shape}"
            )
        
        batch_size, seq_len, hidden_dim = x.shape
        
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Input hidden_dim {hidden_dim} doesn't match expected {self.hidden_dim}"
            )
        
        try:
            # Low-rank projection: (B, T, D) -> (B, T, rank) -> (B, T, H*K)
            h = self.gate_down(x)
            
            # Apply activation and dropout
            h = F.silu(h)  # Swish activation for smooth gradients
            
            if self.dropout > 0.0 and self.training:
                h = self.dropout_layer(h)
            
            # Project back to per-channel gates
            gates = self.gate_up(h)
            
            # Reshape to (batch, seq_len, num_heads, head_dim)
            gates = gates.view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            # Apply sigmoid to bound gates in [0, 1]
            # α_t = sigmoid(gates) represents forget rate per channel
            gates = torch.sigmoid(gates)
            
            # Track performance metrics
            if return_timing:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self.forward_time += elapsed_ms
                self.forward_calls += 1
                return gates, elapsed_ms
            
            return gates, None
            
        except RuntimeError as e:
            # Handle CUDA out of memory gracefully
            if "out of memory" in str(e):
                print(f"ERROR: Out of memory in FineGrainedGating")
                print(f"Input shape: {x.shape}")
                print(f"Try reducing batch size or sequence length")
                # Clear cache and re-raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise
        except Exception as e:
            print(f"ERROR in FineGrainedGating.forward: {e}")
            raise
    
    def get_average_time(self) -> float:
        """Get average forward pass time in milliseconds."""
        if self.forward_calls == 0:
            return 0.0
        return self.forward_time / self.forward_calls
    
    def reset_timing(self):
        """Reset timing statistics."""
        self.forward_time = 0.0
        self.forward_calls = 0
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"hidden_dim={self.hidden_dim}, head_dim={self.head_dim}, "
            f"num_heads={self.num_heads}, rank={self.rank}, dropout={self.dropout}"
        )


def test_fine_grained_gating():
    """
    Test the FineGrainedGating module with various inputs.
    
    Tests:
    - Normal operation
    - Boundary conditions (small/large inputs)
    - Error handling
    - Performance timing
    """
    print("Testing FineGrainedGating...")
    
    # Test configuration
    batch_size = 4
    seq_len = 128
    hidden_dim = 512
    head_dim = 64
    num_heads = 8
    
    # Create module
    try:
        gating = FineGrainedGating(
            hidden_dim=hidden_dim,
            head_dim=head_dim,
            num_heads=num_heads,
            rank=head_dim,
            dropout=0.1,
        )
        print("✓ Module created successfully")
    except Exception as e:
        print(f"✗ Module creation failed: {e}")
        return
    
    # Test 1: Normal forward pass
    try:
        x = torch.randn(batch_size, seq_len, hidden_dim)
        gates, timing = gating(x, return_timing=True)
        
        assert gates.shape == (batch_size, seq_len, num_heads, head_dim)
        assert gates.min() >= 0.0 and gates.max() <= 1.0
        print(f"✓ Normal forward pass (time: {timing:.2f}ms)")
    except Exception as e:
        print(f"✗ Normal forward pass failed: {e}")
        return
    
    # Test 2: Boundary condition - very small input
    try:
        x_small = torch.randn(1, 1, hidden_dim) * 1e-6
        gates_small, _ = gating(x_small, return_timing=False)
        print(f"✓ Small input handling (gates range: [{gates_small.min():.4f}, {gates_small.max():.4f}])")
    except Exception as e:
        print(f"✗ Small input handling failed: {e}")
    
    # Test 3: Boundary condition - very large input
    try:
        x_large = torch.randn(1, 1, hidden_dim) * 100
        gates_large, _ = gating(x_large, return_timing=False)
        print(f"✓ Large input handling (gates range: [{gates_large.min():.4f}, {gates_large.max():.4f}])")
    except Exception as e:
        print(f"✗ Large input handling failed: {e}")
    
    # Test 4: Error handling - wrong input shape
    try:
        x_wrong = torch.randn(batch_size, hidden_dim)  # Missing seq_len dimension
        gates_wrong, _ = gating(x_wrong, return_timing=False)
        print("✗ Should have raised ValueError for wrong input shape")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:50]}...")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Test 5: Performance statistics
    avg_time = gating.get_average_time()
    print(f"✓ Average forward time: {avg_time:.2f}ms over {gating.forward_calls} calls")
    
    print("\n" + "="*60)
    print(f"FineGrainedGating tests complete!")
    print(f"Module info: {gating}")
    print("="*60)


if __name__ == "__main__":
    test_fine_grained_gating()
