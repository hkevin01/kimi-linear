"""
Kimi Delta Attention (KDA) implementation.

This module contains the core KDA mechanism with fine-grained gating,
chunkwise parallelization, and hardware-efficient algorithms.
"""

from .gating import FineGrainedGating
from .state_manager import StateManager
from .dplr import DPLRTransition

__all__ = [
    "FineGrainedGating",
    "StateManager",
    "DPLRTransition",
]
