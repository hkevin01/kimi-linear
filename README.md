# Kimi Linear Optimization Project

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**An optimized implementation of the Kimi Linear architecture - a hybrid linear attention mechanism outperforming traditional full attention.**

[Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Benchmarks](#benchmarks) • [Contributing](#contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Performance](#performance)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Development](#development)
- [Benchmarks](#benchmarks)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🔍 Overview

**Kimi Linear** is a groundbreaking hybrid attention architecture that combines the best of both worlds: the efficiency of linear attention and the performance of full attention mechanisms. This implementation focuses on optimization, hardware efficiency, and production deployment.

### What is Kimi Linear?

Kimi Linear introduces **Kimi Delta Attention (KDA)**, a linear attention mechanism with:
- **Fine-grained gating**: Channel-wise decay for precise memory control
- **Hardware-efficient algorithms**: Specialized DPLR variant optimized for modern GPUs
- **Hybrid architecture**: 3:1 KDA-to-MLA ratio for optimal performance/efficiency

### Why Kimi Linear?

- **🚀 6× faster decoding** at 1M token contexts
- **💾 75% KV cache reduction** for long sequences
- **📊 Superior accuracy**: Matches or exceeds full attention on all benchmarks
- **⚡ Linear complexity**: O(n) vs O(n²) for standard attention
- **🎯 Production-ready**: vLLM integration, Docker support

---

## ✨ Key Features

### Core Implementation

- **Kimi Delta Attention (KDA)**
  - Fine-grained channel-wise gating mechanism
  - Hardware-efficient chunkwise parallelization
  - Delta rule learning with online gradient descent
  - Constrained DPLR formulation for numerical stability

- **Hybrid Architecture**
  - 3:1 KDA-to-MLA ratio (configurable)
  - Multi-Head Latent Attention (MLA) for global context
  - No Position Encoding (NoPE) design
  - Seamless integration with existing frameworks

### Optimization

- **CUDA/Triton Kernels**
  - Fused attention kernels
  - Memory-efficient tiling strategies
  - >80% memory bandwidth utilization
  - 2× faster than general DPLR implementations

- **Memory Management**
  - Fixed-size state (constant memory)
  - Efficient buffer reuse
  - Secondary chunking for numerical stability
  - Mixed precision support (FP16, BF16, FP32)

### Testing & Validation

- **Comprehensive Test Suite**
  - Unit tests (>95% coverage target)
  - Synthetic tasks (Palindrome, MQAR, Stack)
  - Integration tests
  - Benchmark framework

- **Performance Profiling**
  - Kernel-level analysis (Nsight Compute)
  - System-level profiling (Nsight Systems)
  - Memory bandwidth monitoring
  - Automated regression testing

---

## 🏗️ Architecture

```
Input Token Embeddings
         ↓
    [KDA Layer 1] ← Fine-grained gating + Delta rule
         ↓
    [KDA Layer 2] ← State update (dk × dv)
         ↓
    [KDA Layer 3] ← Chunkwise parallelization
         ↓
    [MLA Layer 1] ← Global attention (NoPE)
         ↓
    [Repeat 3:1 ratio...]
         ↓
    Output Logits
```

### Component Details

#### Kimi Delta Attention (KDA)

```python
# Simplified KDA forward pass
def kda_forward(q, k, v, alpha, beta, state):
    # Inter-chunk recurrent update
    state_new = (I - beta * k @ k.T) @ diag(alpha) @ state + beta * k @ v.T
    
    # Intra-chunk parallel attention
    output_intra = tril(q @ k.T) @ v
    
    # Inter-chunk output
    output_inter = (q * gamma.exp()) @ state
    
    return output_intra + output_inter, state_new
```

#### Neural Parameterization

- **Input Projections**: Q, K, V via linear layers + short convolution
- **Gating**: Channel-wise forget gate (α), scalar learning rate (β)
- **Output**: Low-rank gating + RMSNorm
- **Normalization**: L2Norm for Q/K, RMSNorm for output

---

## 📈 Performance

### Speed Benchmarks

| Context Length | MLA (ms) | GDN-H (ms) | Kimi Linear (ms) | Speedup |
|----------------|----------|------------|------------------|---------|
| 4K             | 2.1      | 2.0        | 2.0              | 1.05×   |
| 128K           | 45.2     | 18.3       | 11.4             | 3.98×   |
| 512K           | 182.7    | 76.1       | 79.4             | 2.30×   |
| 1M             | 365.4    | 150.2      | 125.8            | 2.90×   |

**Decoding TPOT (Time Per Output Token)**:
- 4K: 1.84ms (Kimi Linear) vs 1.85ms (MLA) = 1.01× speedup
- 1M: 1.84ms (Kimi Linear) vs 11.48ms (MLA) = **6.3× speedup** ⚡

### Memory Efficiency

| Metric | Full Attention (MLA) | Kimi Linear | Reduction |
|--------|----------------------|-------------|-----------|
| KV Cache @ 128K | 16.0 GB | 4.0 GB | **75%** |
| Peak Memory @ 1M | 128.0 GB | 32.0 GB | **75%** |
| State Size per Head | Linear (O(n)) | Constant (dk × dv) | N/A |

### Accuracy Benchmarks

| Task | Context | MLA | GDN-H | Kimi Linear |
|------|---------|-----|-------|-------------|
| MMLU-Pro | 4K | 47.2 | 47.9 | **51.0** ✅ |
| RULER | 128K | 81.3 | 80.5 | **84.3** ✅ |
| MATH500 | 4K | 80.8 | 83.0 | **81.2** |
| AIME 2025 | 4K | 20.6 | 21.1 | **21.3** ✅ |

---

## 🚀 Installation

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.6
- CUDA >= 12.0 (for GPU acceleration)
- fla-core >= 0.4.0

### Option 1: From Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/kimi-linear.git
cd kimi-linear

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Option 2: Using Docker

```bash
# Build the Docker image
docker build -t kimi-linear:latest -f docker/Dockerfile .

# Run the container
docker run --gpus all -it kimi-linear:latest
```

### Option 3: Using pip (Future)

```bash
# Once published to PyPI
pip install kimi-linear
```

---

## 🎯 Quick Start

### Basic Usage

```python
import torch
from kimi_linear import KimiLinearAttention

# Initialize model
model = KimiLinearAttention(
    dim=1024,
    num_heads=16,
    head_dim=128,
    hybrid_ratio=3,  # 3 KDA layers per 1 MLA layer
)

# Forward pass
x = torch.randn(1, 4096, 1024)  # (batch, seq_len, dim)
output = model(x)

print(f"Output shape: {output.shape}")  # (1, 4096, 1024)
```

### Inference with Pre-trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain Kimi Linear in simple terms."}
]

input_ids = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True, 
    return_tensors="pt"
).to(model.device)

generated_ids = model.generate(inputs=input_ids, max_new_tokens=500)
response = tokenizer.batch_decode(generated_ids)[0]
print(response)
```

### Running Benchmarks

```bash
# Run all benchmarks
python scripts/benchmark/run_benchmarks.py --model kimi-linear --baseline mla

# Run specific benchmark
python scripts/benchmark/run_benchmarks.py --task mmlu-pro --context-length 4096

# Profile performance
python scripts/profiling/profile_attention.py --kernel kda --chunk-size 64
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest tests/unit/

# Run with coverage
pytest --cov=src --cov-report=html tests/

# Run synthetic tasks
python tests/synthetic/test_palindrome.py
python tests/synthetic/test_mqar.py
python tests/synthetic/test_stack.py
```

---

## 📂 Project Structure

```
kimi-linear/
├── src/                          # Source code
│   ├── kda/                      # Kimi Delta Attention implementation
│   │   ├── gating.py            # Fine-grained gating mechanism
│   │   ├── state_manager.py    # State tracking and updates
│   │   ├── wy_representation.py # WY representation for rank-1 updates
│   │   ├── ut_transform.py      # UT transform
│   │   ├── chunk_update.py      # Chunkwise state updates
│   │   └── dplr.py              # DPLR variant implementation
│   ├── attention/                # Attention mechanisms
│   │   ├── linear_attention.py  # Base linear attention
│   │   ├── delta_rule.py        # Delta rule learning
│   │   └── mla.py               # Multi-Head Latent Attention
│   ├── models/                   # Model architectures
│   │   ├── kimi_linear.py       # Hybrid Kimi Linear model
│   │   ├── projections.py       # Input projections
│   │   ├── conv_layer.py        # Short convolution
│   │   └── gating.py            # Output/forget gates
│   ├── kernels/                  # Optimized kernels
│   │   ├── kda_fused_kernel.cu  # CUDA implementation
│   │   └── kda_triton.py        # Triton implementation
│   ├── utils/                    # Utility functions
│   │   ├── performance_logger.py
│   │   └── memory_monitor.py
│   └── benchmarks/               # Benchmark utilities
├── tests/                        # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── synthetic/               # Synthetic task tests
├── scripts/                      # Scripts
│   ├── setup/                   # Setup scripts
│   ├── benchmark/               # Benchmarking scripts
│   └── profiling/               # Profiling tools
├── docs/                         # Documentation
│   ├── api/                     # API documentation
│   ├── tutorials/               # Tutorials and guides
│   ├── architecture/            # Architecture docs
│   └── project-plan.md          # Comprehensive project plan
├── data/                         # Data directory
│   ├── synthetic/               # Synthetic test data
│   ├── benchmarks/              # Benchmark results
│   └── results/                 # Experimental results
├── assets/                       # Assets (figures, diagrams)
├── docker/                       # Docker configurations
├── .github/                      # GitHub-specific files
│   └── workflows/               # CI/CD workflows
├── .copilot/                     # Copilot configurations
├── .vscode/                      # VS Code settings
├── memory-bank/                  # Memory bank system
│   ├── app-description.md       # Project description
│   ├── change-log.md            # Change log
│   └── implementation-plans/    # Implementation plans
├── configs/                      # Configuration files
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── pyproject.toml               # Project metadata
├── .gitignore                   # Git ignore rules
├── .editorconfig                # Editor configuration
├── LICENSE                      # MIT License
└── README.md                    # This file
```

---

## 🛠️ Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/

# Run linting
pylint src/
flake8 src/

# Run type checking
mypy src/
```

### Building Documentation

```bash
cd docs/
make html
# Documentation will be in docs/_build/html/
```

### Running in Docker (Development)

```bash
# Build development image
docker build -t kimi-linear:dev -f docker/Dockerfile.dev .

# Run with GPU and mounted source
docker run --gpus all -v $(pwd):/workspace -it kimi-linear:dev bash
```

### Code Style Guidelines

- **Python**: Follow PEP 8, use Black formatter (88 char line length)
- **C++**: Follow Google C++ Style Guide (100 char line length)
- **Java**: Follow Google Java Style Guide
- **Naming Conventions**:
  - Functions/methods: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private members: `_leading_underscore`

---

## 📊 Benchmarks

### Running Comprehensive Benchmarks

```bash
# Full benchmark suite (requires GPU with 24GB+ VRAM)
python scripts/benchmark/run_benchmarks.py \
    --models kimi-linear mla gdn-h \
    --context-lengths 4096 32768 131072 524288 1048576 \
    --tasks all \
    --output-dir data/benchmarks/results

# Quick benchmark (lighter tests)
python scripts/benchmark/run_benchmarks.py \
    --models kimi-linear mla \
    --context-lengths 4096 32768 \
    --tasks mmlu-pro ruler \
    --quick
```

### Synthetic Task Evaluation

```bash
# Palindrome test (sequence reversal)
python tests/synthetic/test_palindrome.py --lengths 256 512 1024 2048

# MQAR test (associative recall)
python tests/synthetic/test_mqar.py --num-queries 5 10 20

# Stack test (state tracking)
python tests/synthetic/test_stack.py --num-stacks 64 --sequence-length 1024
```

### Performance Profiling

```bash
# Kernel profiling with Nsight Compute
ncu --set full python scripts/profiling/profile_kernels.py

# System profiling with Nsight Systems
nsys profile -o profile.nsys-rep python scripts/profiling/profile_system.py

# Memory profiling
python scripts/profiling/profile_memory.py --max-context 1048576
```

---

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Quick Start Guide](docs/tutorials/quickstart.md)**: Get started in 5 minutes
- **[API Reference](docs/api/)**: Complete API documentation
- **[Architecture Guide](docs/architecture/)**: Deep dive into the architecture
- **[Training Guide](docs/tutorials/training.md)**: Training your own models
- **[Advanced Usage](docs/tutorials/advanced.md)**: Custom kernels and optimizations
- **[Project Plan](docs/project-plan.md)**: Comprehensive development roadmap

### Additional Resources

- **Research Paper**: [Kimi Linear Technical Report](https://arxiv.org/abs/2510.26692)
- **Original Implementation**: [MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)
- **FLA Kernels**: [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- **Pre-trained Models**: [HuggingFace Hub](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](.github/CONTRIBUTING.md) for details.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Run tests** (`pytest tests/`)
5. **Commit your changes** (`git commit -m 'Add amazing feature'`)
6. **Push to branch** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

### Development Priorities

- 🔴 **Critical**: Core functionality, bug fixes, performance regressions
- 🟠 **High**: New features, optimizations
- 🟡 **Medium**: Documentation improvements, refactoring
- 🟢 **Low**: Code style, minor enhancements

---

## �� Citation

If you use Kimi Linear in your research, please cite:

```bibtex
@misc{team2025kimi,
  title         = {Kimi Linear: An Expressive, Efficient Attention Architecture},
  author        = {Zhang, Yu and Lin, Zongyu and Yao, Xingcheng and Hu, Jiaxi and others},
  year          = {2025},
  eprint        = {2510.26692},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Moonshot AI** for the original Kimi Linear research and implementation
- **FLA Team** for the flash linear attention kernels
- **DeepSeek** for MLA architecture insights
- **Community contributors** for feedback and improvements

---

## 📬 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/kimi-linear/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/kimi-linear/discussions)

---

<div align="center">

**⭐ Star this repository if you find it useful!**

Made with ❤️ by the Kimi Linear Optimization Team

</div>
