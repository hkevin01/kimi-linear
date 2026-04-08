"""
Kimi Linear: An Expressive, Efficient Attention Architecture

This package implements the Kimi Linear hybrid attention mechanism with
hardware-efficient kernels and comprehensive benchmarking.
"""

__version__ = "0.1.0"
__author__ = "Kimi Linear Optimization Team"

from .kda import (
    FineGrainedGating,
    StateManager,
    DPLRTransition,
    KDALayer,
    ChunkwiseParallelKDA,
    MLALayer,
    KDAVLLMAdapter,
    KDAStateBlockManager,
    chunk_kda_forward,
    fused_recurrent_kda_forward,
    HAS_TRITON,
)

__all__ = [
    "FineGrainedGating",
    "StateManager",
    "DPLRTransition",
    "KDALayer",
    "ChunkwiseParallelKDA",
    "MLALayer",
    "KDAVLLMAdapter",
    "KDAStateBlockManager",
    "chunk_kda_forward",
    "fused_recurrent_kda_forward",
    "HAS_TRITON",
]
