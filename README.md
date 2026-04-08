# Kimi Linear

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.26692-b31b1b.svg)](https://arxiv.org/abs/2510.26692)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**An open-source implementation of the Kimi Linear hybrid linear attention architecture.**

[Installation](#installation) · [Quick Start](#quick-start) · [Architecture](#architecture) · [Benchmarks](#benchmarks) · [Contributing](#contributing)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Benchmarks](#benchmarks)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

---

## Overview

Kimi Linear is a hybrid attention architecture that combines linear attention with full attention in a layered approach. At its core is **Kimi Delta Attention (KDA)**, a linear attention module that extends Gated DeltaNet with channel-wise gating, enabling more effective use of finite-state RNN memory compared to head-wise gating approaches.

The architecture uses a **3:1 KDA-to-MLA (Multi-Head Latent Attention) ratio**, placing three KDA layers for every one full attention layer. This hybrid approach provides significant efficiency improvements while maintaining competitive accuracy on long-context tasks.

Based on the [Kimi Linear technical report](https://arxiv.org/abs/2510.26692) by the Kimi Team.

### Key Results (from original paper)

| Metric | Value |
|--------|-------|
| KV cache reduction | Up to 75% |
| Decoding throughput (1M context) | Up to 6× vs full MLA |
| Prefill speedup (512k–1M tokens) | 2.3–2.9× vs MLA |
| MMLU-Pro (4k) | ≥ 51.0 |

---

## Features

- **Kimi Delta Attention (KDA)** — channel-wise decay gates for fine-grained memory control
- **DPLR transition matrices** — specialized Diagonal-Plus-Low-Rank formulation, 2× faster than general DPLR
- **Recurrent state management** — constant memory O(dk × dv) regardless of sequence length
- **Chunkwise parallelization** — inter-chunk recurrent + intra-chunk parallel strategy
- **Mixed precision support** — FP16, BF16, and FP32
- **Numerical stability** — NaN/Inf detection, eigenvalue monitoring, secondary chunking

---

## Architecture

### KDA State Update

At each step, the state $S_t \in \mathbb{R}^{d_k \times d_v}$ is updated as:

$$S_t = \bigl(\text{Diag}(\alpha_t) - \beta_t k_t k_t^\top \text{Diag}(\alpha_t)\bigr) S_{t-1} + \beta_t k_t v_t^\top$$

This is computed in two steps for efficiency:

1. **Diagonal decay**: $S' = \text{Diag}(\alpha_t) \cdot S_{t-1}$
2. **Rank-1 correction**: $S_t = (I - \beta_t k_t k_t^\top) S'$

### Hybrid Layer Stack

```
Input → [KDA] → [KDA] → [KDA] → [MLA] → (repeat, 3:1 ratio) → Output
```

### Component Summary

| Component | File | Time Complexity | Space Complexity |
|-----------|------|----------------|-----------------|
| `FineGrainedGating` | `src/kda/gating.py` | O(B·T·D·rank) | O(D·rank) |
| `StateManager` | `src/kda/state_manager.py` | O(B·H·K·V) | O(B·H·K·V) |
| `DPLRTransition` | `src/kda/dplr.py` | O(B·H·K·V) | O(B·H·K·V) |
| `KDALayer` | `src/kda/kda_layer.py` | O(B·T·K·V) | O(B·H·K·V) |

---

## Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.6
- CUDA >= 12.0 (for GPU support)

### From source

```bash
git clone https://github.com/hkevin01/kimi-linear.git
cd kimi-linear
pip install -e ".[dev]"
```

### Dependencies only

```bash
pip install -r requirements.txt
```

### Docker

```bash
# Development environment
docker build -f docker/Dockerfile.dev -t kimi-linear:dev .
docker run --gpus all -it kimi-linear:dev

# Production environment
docker build -f docker/Dockerfile -t kimi-linear:latest .
```

---

## Quick Start

```python
import torch
from src.kda import FineGrainedGating, StateManager, DPLRTransition
from src.kda.kda_layer import KDALayer

# Configure dimensions
batch_size = 4
seq_len = 128
hidden_dim = 512
num_heads = 8
head_dim = 64

# Create a single KDA layer
layer = KDALayer(
    hidden_dim=hidden_dim,
    head_dim=head_dim,
    num_heads=num_heads,
)

# Forward pass
x = torch.randn(batch_size, seq_len, hidden_dim)
output, state = layer(x)
print(f"Output shape: {output.shape}")  # (4, 128, 512)

# Use individual components
gating = FineGrainedGating(hidden_dim=512, head_dim=64, num_heads=8)
state_mgr = StateManager(key_dim=64, value_dim=64, num_heads=8)
dplr = DPLRTransition(key_dim=64, value_dim=64, num_heads=8)

gates, _ = gating(x)
state = state_mgr.initialize_state(batch_size)
```

---

## Project Structure

```
kimi-linear/
├── src/
│   └── kda/
│       ├── __init__.py          # Package exports
│       ├── gating.py            # FineGrainedGating: channel-wise decay gates
│       ├── state_manager.py     # StateManager: recurrent state with checkpointing
│       ├── dplr.py              # DPLRTransition: specialized DPLR update
│       └── kda_layer.py         # KDALayer: assembled KDA module
├── tests/
│   └── kda/
│       ├── test_gating.py       # Tests for FineGrainedGating
│       ├── test_state_manager.py
│       └── test_dplr.py
├── scripts/
│   └── benchmark/
│       └── run_benchmarks.py    # Performance benchmarks
├── docs/
│   ├── project-plan.md
│   ├── PROJECT_STATUS.md
│   └── IMPLEMENTATION_SUMMARY.md
├── docker/
│   ├── Dockerfile
│   └── Dockerfile.dev
├── memory-bank/
│   └── app-description.md
├── requirements.txt
├── setup.py
└── README.md
```

---

## Benchmarks

Run the benchmark suite:

```bash
python scripts/benchmark/run_benchmarks.py
```

Run tests:

```bash
pytest -v tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

---

## Development

### Code Style

This project uses [Black](https://github.com/psf/black) for formatting and [pylint](https://pylint.org/) for linting.

```bash
black src/ tests/
pylint src/
```

### Type Checking

```bash
mypy src/
```

### Adding a New Component

1. Create module in `src/kda/` or relevant subpackage
2. Add tests in `tests/kda/`
3. Export from `src/kda/__init__.py`

---

## Implementation Status

| Component | Status |
|-----------|--------|
| `FineGrainedGating` | ✅ Complete |
| `StateManager` | ✅ Complete |
| `DPLRTransition` | ✅ Complete |
| `KDALayer` | ✅ Complete |
| Chunkwise parallelization | ⭕ Planned |
| WY representation | ⭕ Planned |
| UT transform | ⭕ Planned |
| MLA integration | ⭕ Planned |
| CUDA/Triton kernels | ⭕ Planned |
| vLLM integration | ⭕ Planned |

---

## Citation

If this implementation is useful in your work, please cite the original paper:

```bibtex
@article{kimiteam2025kimilinear,
  title={Kimi Linear: An Expressive, Efficient Attention Architecture},
  author={Kimi Team},
  journal={arXiv preprint arXiv:2510.26692},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The Kimi Linear architecture is described in [arXiv:2510.26692](https://arxiv.org/abs/2510.26692).
