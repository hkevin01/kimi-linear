---
applyTo: '**'
---

# Kimi Linear Optimization Project

## Project Overview

This project implements and optimizes the **Kimi Linear** architecture - a groundbreaking hybrid linear attention mechanism that outperforms traditional full attention methods while maintaining superior hardware efficiency.

## Core Features

### 1. Kimi Delta Attention (KDA)
- **Fine-grained gating mechanism**: Channel-wise decay for precise memory control
- **Hardware-efficient chunkwise algorithm**: Specialized DPLR variant for optimal performance
- **Delta rule learning**: Online gradient descent on reconstruction objectives

### 2. Hybrid Architecture
- **3:1 KDA-to-MLA ratio**: Optimal balance between performance and efficiency
- **Multi-Head Latent Attention (MLA)**: Global attention layers for long-range dependencies
- **NoPE design**: Position-aware through KDA, simplifying global attention

### 3. Efficiency Gains
- **75% KV cache reduction**: Significantly lower memory footprint
- **6× faster decoding**: For 1M token contexts
- **Linear time complexity**: O(n) vs O(n²) for standard attention
- **Constant memory inference**: Fixed-size state regardless of sequence length

## Technical Stack

### Primary Languages
- **Python**: Core implementation, training, and evaluation
- **C++/CUDA**: Hardware-optimized kernels and operators
- **Triton**: GPU kernel optimization

### Key Dependencies
- PyTorch >= 2.6
- fla-core >= 0.4.0 (Flash Linear Attention library)
- transformers (Hugging Face)
- numpy, scipy (numerical computing)
- pytest (testing framework)

## Target Users

1. **ML Researchers**: Exploring efficient attention mechanisms
2. **LLM Developers**: Building scalable language models
3. **Performance Engineers**: Optimizing inference pipelines
4. **Academic Institutions**: Studying attention architectures

## Project Goals

### Short-term Goals
1. Implement core KDA mechanism with chunkwise parallelization
2. Create comprehensive test suites (synthetic and real-world)
3. Build benchmarking infrastructure for performance comparison
4. Document API and provide usage examples

### Long-term Goals
1. Achieve parity with full attention on all benchmarks
2. Optimize for multiple hardware backends (CUDA, ROCm, TPU)
3. Integrate with popular frameworks (vLLM, TGI, llama.cpp)
4. Scale to billion-parameter models with 1M+ context lengths
5. Contribute optimizations back to the open-source community

## Key Innovations

### 1. Mathematical Foundation
- Generalized DPLR (Diagonal-Plus-Low-Rank) structure
- Fine-grained multiplicative positional encoding
- Weight decay on fast weights for memory management

### 2. Algorithmic Efficiency
- WY representation for rank-1 updates
- UT transform for reduced non-matmul FLOPs
- Secondary chunking for numerical stability

### 3. System Design
- Inter-chunk recurrent, intra-chunk parallel strategy
- Seamless integration with existing caching systems
- Drop-in replacement for full attention

## Performance Targets

### Accuracy
- **MMLU-Pro (4k)**: ≥ 51.0 (matching or exceeding full attention)
- **RULER (128k)**: ≥ 84.3 with 4× speedup
- **Long-context benchmarks**: Pareto-optimal performance/efficiency trade-off

### Efficiency
- **Prefilling**: 2.3-2.9× faster than MLA for 512k-1M tokens
- **Decoding**: 6.3× faster TPOT at 1M context length
- **Memory**: 75% reduction in KV cache usage

## Architecture Highlights

```
Input → [KDA Layer] → [KDA Layer] → [KDA Layer] → [MLA Layer] → Repeat (3:1 ratio)
         ↓             ↓             ↓             ↓
      State Update   State Update  State Update  Global Attention
```

### Components
- **KDA Layers**: Linear attention with delta rule and fine-grained gating
- **MLA Layers**: Multi-Head Latent Attention for global context
- **MoE**: Mixture of Experts for efficient capacity scaling
- **No Position Encoding**: Position awareness through KDA mechanism

## Development Principles

1. **Performance First**: Every optimization must show measurable improvement
2. **Mathematical Rigor**: Maintain theoretical soundness
3. **Hardware Awareness**: Design with GPU/TPU architecture in mind
4. **Reproducibility**: All experiments must be reproducible
5. **Community Driven**: Open-source, well-documented, accessible

## Competitive Advantages

1. **First to outperform full attention**: Across short, long, and RL contexts
2. **Superior scaling**: Efficient for sequences up to 1M tokens
3. **Production-ready**: vLLM integration, Docker support
4. **Comprehensive evaluation**: 30+ benchmarks across multiple domains
5. **Open-source**: MIT license, full kernel and model release

## Use Cases

- **Long-document understanding**: Legal, medical, research papers
- **Code repositories**: Multi-file context understanding
- **Conversational AI**: Extended dialogue history
- **Agentic workflows**: Tool use, planning, reasoning
- **RL applications**: Test-time scaling, policy optimization

## Success Metrics

1. **Benchmark Performance**: Match or exceed full attention on all tasks
2. **Inference Speed**: Achieve target speedups (2-6× depending on context)
3. **Memory Efficiency**: 75% KV cache reduction validated
4. **Adoption**: Community usage, citations, integrations
5. **Stability**: Robust error handling, graceful degradation

---

**Version**: 1.0.0
**Last Updated**: 2025-01-XX
**Status**: Active Development
