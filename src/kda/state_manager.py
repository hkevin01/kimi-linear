"""
State management for Kimi Delta Attention.

Handles the matrix-valued recurrent state that accumulates
key-value associations with finite-state RNN memory.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import time
import warnings


class StateManager(nn.Module):
    """
    Manages recurrent state for KDA layers.

    The state St ∈ R^(dk × dv) accumulates key-value associations
    with forgetting mechanism. Maintains constant memory regardless
    of sequence length.

    Args:
        key_dim: Dimension of keys (dk)
        value_dim: Dimension of values (dv)
        num_heads: Number of attention heads
        max_batch_size: Maximum batch size for pre-allocation
        dtype: Data type for state (default: torch.float32)
        device: Device for state (default: 'cpu')

    Memory Usage: O(batch_size * num_heads * key_dim * value_dim)
    """

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        num_heads: int,
        max_batch_size: int = 32,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu',
    ):
        super().__init__()

        # Validate parameters
        if key_dim <= 0 or value_dim <= 0:
            raise ValueError(f"Dimensions must be positive: key_dim={key_dim}, value_dim={value_dim}")
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

        # Pre-allocate state buffer for efficiency
        # Shape: (max_batch_size, num_heads, key_dim, value_dim)
        self.register_buffer(
            'state_buffer',
            torch.zeros(
                max_batch_size, num_heads, key_dim, value_dim,
                dtype=dtype, device=device
            )
        )

        # Track active batch size
        self.current_batch_size = 0

        # Performance metrics
        self.update_time = 0.0
        self.update_calls = 0
        self.memory_allocated = 0

        # State persistence for long sequences
        self.enable_checkpointing = False
        self.checkpoint_interval = 1000  # Save state every N steps
        self.checkpoints: Dict[int, torch.Tensor] = {}

    def initialize_state(
        self,
        batch_size: int,
        initial_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Initialize or reset the state.

        Args:
            batch_size: Current batch size
            initial_state: Optional initial state to load
                          Shape: (batch_size, num_heads, key_dim, value_dim)

        Returns:
            Initialized state tensor

        Raises:
            ValueError: If batch_size exceeds max_batch_size
            RuntimeError: If initialization fails
        """
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"batch_size {batch_size} exceeds max_batch_size {self.max_batch_size}. "
                f"Consider increasing max_batch_size or reducing batch size."
            )

        try:
            self.current_batch_size = batch_size

            if initial_state is not None:
                # Validate initial state shape
                expected_shape = (batch_size, self.num_heads, self.key_dim, self.value_dim)
                if initial_state.shape != expected_shape:
                    raise ValueError(
                        f"initial_state shape {initial_state.shape} doesn't match "
                        f"expected shape {expected_shape}"
                    )

                # Copy initial state into buffer
                self.state_buffer[:batch_size] = initial_state.to(
                    dtype=self.dtype, device=self.device
                )
            else:
                # Zero initialize
                self.state_buffer[:batch_size].zero_()

            # Track memory usage
            self.memory_allocated = (
                batch_size * self.num_heads * self.key_dim * self.value_dim *
                torch.finfo(self.dtype).bits // 8
            )

            return self.state_buffer[:batch_size].clone()

        except Exception as e:
            print(f"ERROR in initialize_state: {e}")
            raise RuntimeError(f"Failed to initialize state: {e}")

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
        """
        Update state using KDA update rule.

        Implements: St = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T

        Args:
            state: Current state (B, H, K, V)
            keys: Keys (B, H, K)
            values: Values (B, H, V)
            gates: Forget gates α_t (B, H, K)
            beta: Learning rate β_t (B, H, 1)
            step: Current timestep for checkpointing
            return_timing: If True, return execution time

        Returns:
            new_state: Updated state (B, H, K, V)
            timing: Execution time in milliseconds (if return_timing)

        Time Complexity: O(B * H * K * V)
        Space Complexity: O(B * H * K * V)
        """
        start_time = time.perf_counter() if return_timing else None

        # Validate shapes
        B, H, K, V = state.shape
        if keys.shape != (B, H, K):
            raise ValueError(f"keys shape {keys.shape} doesn't match expected {(B, H, K)}")
        if values.shape != (B, H, V):
            raise ValueError(f"values shape {values.shape} doesn't match expected {(B, H, V)}")
        if gates.shape != (B, H, K):
            raise ValueError(f"gates shape {gates.shape} doesn't match expected {(B, H, K)}")

        try:
            # Apply fine-grained decay: S_decayed = Diag(α_t) S_{t-1}
            # Broadcasting: gates (B, H, K, 1) * state (B, H, K, V)
            gates_expanded = gates.unsqueeze(-1)  # (B, H, K, 1)
            state_decayed = gates_expanded * state  # Element-wise decay per channel

            # Compute delta rule correction: -β_t k_t k_t^T S_decayed
            # k_t k_t^T S_decayed = k_t (k_t^T S_decayed)
            keys_expanded = keys.unsqueeze(-1)  # (B, H, K, 1)
            kt_state = torch.einsum('bhk,bhkv->bhv', keys, state_decayed)  # (B, H, V)
            correction = torch.einsum('bhk,bhv->bhkv', keys, kt_state)  # (B, H, K, V)
            # Expand beta from (B, H, 1) to (B, H, 1, 1) for broadcasting
            beta_expanded = beta.unsqueeze(-1) if beta.dim() == 3 else beta.unsqueeze(-1).unsqueeze(-1)
            correction = beta_expanded * correction  # Scale by β_t

            # Add new key-value association: β_t k_t v_t^T
            kv_update = torch.einsum('bhk,bhv->bhkv', keys, values)  # (B, H, K, V)
            kv_update = beta_expanded * kv_update

            # Final update: St = S_decayed - correction + kv_update
            new_state = state_decayed - correction + kv_update

            # Check for numerical issues
            if torch.isnan(new_state).any():
                warnings.warn("NaN detected in state update! Resetting to previous state.")
                new_state = state.clone()
            elif torch.isinf(new_state).any():
                warnings.warn("Inf detected in state update! Clipping values.")
                new_state = torch.clamp(new_state, -1e6, 1e6)

            # Checkpoint state if enabled
            if self.enable_checkpointing and step % self.checkpoint_interval == 0:
                self.checkpoints[step] = new_state.detach().clone()
                # Limit checkpoint storage to prevent memory issues
                if len(self.checkpoints) > 10:
                    oldest_step = min(self.checkpoints.keys())
                    del self.checkpoints[oldest_step]

            # Update metrics
            if return_timing:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self.update_time += elapsed_ms
                self.update_calls += 1
                return new_state, elapsed_ms

            return new_state, None

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"ERROR: Out of memory in state update")
                print(f"State shape: {state.shape}, Memory allocated: {self.memory_allocated} bytes")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise
        except Exception as e:
            print(f"ERROR in update_state: {e}")
            raise

    def load_checkpoint(self, step: int) -> Optional[torch.Tensor]:
        """Load state from checkpoint."""
        return self.checkpoints.get(step)

    def clear_checkpoints(self):
        """Clear all checkpoints to free memory."""
        self.checkpoints.clear()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics in MB."""
        buffer_size_mb = (
            self.state_buffer.numel() * self.state_buffer.element_size() / 1024 / 1024
        )
        checkpoint_size_mb = sum(
            ckpt.numel() * ckpt.element_size() / 1024 / 1024
            for ckpt in self.checkpoints.values()
        )
        return {
            'buffer_mb': buffer_size_mb,
            'checkpoints_mb': checkpoint_size_mb,
            'total_mb': buffer_size_mb + checkpoint_size_mb,
        }

    def get_average_update_time(self) -> float:
        """Get average update time in milliseconds."""
        if self.update_calls == 0:
            return 0.0
        return self.update_time / self.update_calls

    def reset_timing(self):
        """Reset timing statistics."""
        self.update_time = 0.0
        self.update_calls = 0

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"key_dim={self.key_dim}, value_dim={self.value_dim}, "
            f"num_heads={self.num_heads}, max_batch_size={self.max_batch_size}, "
            f"dtype={self.dtype}"
        )


def test_state_manager():
    """Test StateManager functionality."""
    print("Testing StateManager...")

    # Configuration
    batch_size = 4
    num_heads = 8
    key_dim = 64
    value_dim = 64

    # Create manager
    manager = StateManager(
        key_dim=key_dim,
        value_dim=value_dim,
        num_heads=num_heads,
        max_batch_size=32,
    )

    # Test 1: Initialize state
    try:
        state = manager.initialize_state(batch_size)
        assert state.shape == (batch_size, num_heads, key_dim, value_dim)
        print(f"✓ State initialization (shape: {state.shape})")
    except Exception as e:
        print(f"✗ State initialization failed: {e}")
        return

    # Test 2: Update state
    try:
        keys = torch.randn(batch_size, num_heads, key_dim) / (key_dim ** 0.5)  # Normalized keys
        values = torch.randn(batch_size, num_heads, value_dim)
        gates = torch.sigmoid(torch.randn(batch_size, num_heads, key_dim))
        beta = torch.sigmoid(torch.randn(batch_size, num_heads, 1))

        new_state, timing = manager.update_state(
            state, keys, values, gates, beta,
            step=0, return_timing=True
        )

        assert new_state.shape == state.shape
        print(f"✓ State update (time: {timing:.2f}ms)")
    except Exception as e:
        print(f"✗ State update failed: {e}")
        return

    # Test 3: Memory usage
    memory_info = manager.get_memory_usage()
    print(f"✓ Memory usage: {memory_info['total_mb']:.2f} MB")

    # Test 4: Checkpointing
    manager.enable_checkpointing = True
    manager.checkpoint_interval = 1

    for step in range(5):
        state, _ = manager.update_state(
            state, keys, values, gates, beta, step=step
        )

    assert len(manager.checkpoints) > 0
    print(f"✓ Checkpointing ({len(manager.checkpoints)} checkpoints saved)")

    # Test 5: Error handling - batch size too large
    try:
        large_state = manager.initialize_state(100)  # Exceeds max_batch_size
        print("✗ Should have raised ValueError for large batch size")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:50]}...")

    print(f"\n{'='*60}")
    print(f"StateManager tests complete!")
    print(f"Module info: {manager}")
    print(f"Average update time: {manager.get_average_update_time():.2f}ms")
    print('='*60)


if __name__ == "__main__":
    test_state_manager()
