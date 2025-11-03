"""
Kimi Linear: An Expressive, Efficient Attention Architecture

This package implements the Kimi Linear hybrid attention mechanism with
hardware-efficient kernels and comprehensive benchmarking.
"""

__version__ = "0.1.0"
__author__ = "Kimi Linear Optimization Team"

from src.kda import KimiDeltaAttention
from src.attention import LinearAttention, DeltaRule
from src.models import KimiLinearModel

__all__ = [
    "KimiDeltaAttention",
    "LinearAttention",
    "DeltaRule", 
    "KimiLinearModel",
]
