from setuptools import setup, find_packages

setup(
    name="kimi-linear",
    version="0.1.0",
    author="Kimi Linear Optimization Team",
    description="Kimi Linear hybrid KDA/MLA attention — chunkwise parallel, vLLM-ready",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hkevin01/kimi-linear",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.6.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "fla": [
            "flash-linear-attention>=0.4.0",  # Triton kernels (optional)
        ],
        "vllm": [
            "vllm>=0.4.0",                    # vLLM deployment (optional)
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "pylint>=3.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
)
