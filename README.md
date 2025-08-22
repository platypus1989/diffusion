# Diffusion Two Moons

A minimal PyTorch implementation of DDPM-style diffusion model for the two moons distribution.

## Quick Setup Guide

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

### Usage

#### Basic Training

Train a diffusion model with default parameters:
```bash
uv run src/run.py
```

#### Custom Training

Train with custom parameters:
```bash
uv run src/run.py --hidden 256 --num-layers 3 --epochs 100 --batch-size 512
```

#### Run Experiments

Use the provided script to run multiple experiments:
```bash
uv run train.sh
```