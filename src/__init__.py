"""
Kimi Linear: An Expressive, Efficient Attention Architecture

This package implements the Kimi Linear hybrid attention mechanism with
hardware-efficient kernels and comprehensive benchmarking.
"""

__version__ = "0.1.0"
__author__ = "Kimi Linear Optimization Team"

from .kda import FineGrainedGating, StateManager, DPLRTransition
from .kda.kda_layer import KDALayer

__all__ = [
    "FineGrainedGating",
    "StateManager",
    "DPLRTransition",
    "KDALayer",
]
