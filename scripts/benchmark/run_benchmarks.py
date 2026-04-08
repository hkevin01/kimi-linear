"""
Benchmark script for Kimi Linear KDA components.

Measures forward pass latency and throughput for FineGrainedGating,
StateManager, DPLRTransition, and the assembled KDALayer.

Usage:
    python scripts/benchmark/run_benchmarks.py
    python scripts/benchmark/run_benchmarks.py --seq-len 512 --batch-size 8
"""

import argparse
import time
import sys
import os

# Add project root to path so src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F

from src.kda.gating import FineGrainedGating
from src.kda.state_manager import StateManager
from src.kda.dplr import DPLRTransition
from src.kda.kda_layer import KDALayer


def bench(fn, warmup=5, runs=20):
    """Run fn warmup+runs times and return mean latency (ms) over runs."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / runs
    return elapsed


def benchmark_gating(hidden_dim, head_dim, num_heads, batch_size, seq_len, device):
    module = FineGrainedGating(hidden_dim, head_dim, num_heads).to(device)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    ms = bench(lambda: module(x))
    print(f"  FineGrainedGating   B={batch_size} T={seq_len} D={hidden_dim}  →  {ms:.2f} ms/iter")


def benchmark_state_update(key_dim, value_dim, num_heads, batch_size, device):
    manager = StateManager(key_dim, value_dim, num_heads, max_batch_size=batch_size + 4).to(device)
    dplr = DPLRTransition(key_dim, value_dim, num_heads).to(device)
    state = manager.initialize_state(batch_size).to(device)
    keys = F.normalize(torch.randn(batch_size, num_heads, key_dim, device=device), dim=-1)
    values = torch.randn(batch_size, num_heads, value_dim, device=device)
    gates = torch.sigmoid(torch.randn(batch_size, num_heads, key_dim, device=device))
    beta = torch.sigmoid(torch.randn(batch_size, num_heads, 1, device=device)) * 0.5
    ms = bench(lambda: dplr(state, keys, values, gates, beta))
    print(f"  DPLR single step    B={batch_size} H={num_heads} K={key_dim}   →  {ms:.2f} ms/iter")


def benchmark_kda_layer(hidden_dim, head_dim, num_heads, batch_size, seq_len, device):
    layer = KDALayer(
        hidden_dim=hidden_dim,
        head_dim=head_dim,
        num_heads=num_heads,
        max_batch_size=batch_size + 4,
    ).to(device)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    ms = bench(lambda: layer(x))
    throughput = (batch_size * seq_len) / (ms / 1000)
    print(
        f"  KDALayer (full seq) B={batch_size} T={seq_len} D={hidden_dim}  →  "
        f"{ms:.2f} ms/iter  ({throughput/1e3:.1f}k tokens/s)"
    )


def main():
    parser = argparse.ArgumentParser(description="Kimi Linear benchmarks")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"\nKimi Linear Benchmark")
    print(f"Device: {args.device}")
    print(f"Config: batch={args.batch_size}, seq_len={args.seq_len}, hidden={args.hidden_dim}")
    print("-" * 60)

    benchmark_gating(args.hidden_dim, args.head_dim, args.num_heads, args.batch_size, args.seq_len, args.device)
    benchmark_state_update(args.head_dim, args.head_dim, args.num_heads, args.batch_size, args.device)
    benchmark_kda_layer(args.hidden_dim, args.head_dim, args.num_heads, args.batch_size, args.seq_len, args.device)

    print("-" * 60)
    print("Done.\n")


if __name__ == "__main__":
    main()
