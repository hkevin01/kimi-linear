"""
Kimi Delta Attention (KDA) implementation.

This module contains the core KDA mechanism with fine-grained gating,
chunkwise parallelization, and hardware-efficient algorithms.

Components:
    - FineGrainedGating: Channel-wise decay gates for precise memory control
    - StateManager: Recurrent state management with checkpointing
    - DPLRTransition: Specialized DPLR transition matrices for efficient updates

Example usage:
    >>> from src.kda import FineGrainedGating, StateManager, DPLRTransition
    >>>
    >>> # Create components
    >>> gating = FineGrainedGating(hidden_dim=512, head_dim=64, num_heads=8)
    >>> state_mgr = StateManager(key_dim=64, value_dim=64, num_heads=8)
    >>> dplr = DPLRTransition(key_dim=64, value_dim=64, num_heads=8)
    >>>
    >>> # Use in forward pass
    >>> x = torch.randn(batch, seq_len, hidden_dim)
    >>> gates, _ = gating(x)
    >>> state = state_mgr.initialize_state(batch)
    >>> new_state, _ = dplr(state, keys, values, gates, beta)
"""

from .gating import FineGrainedGating
from .state_manager import StateManager
from .dplr import DPLRTransition

__all__ = [
    "FineGrainedGating",
    "StateManager",
    "DPLRTransition",
]

__version__ = "0.1.0"
