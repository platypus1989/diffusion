"""
Diffusion Model Evaluation Framework for 2D Scientific Distributions

This package provides a modular framework for training and evaluating
diffusion models on 2D scientific distributions.
"""

from .data import TwoMoonsDataset
from .model import DiffusionMLP, create_model
from .train import train_diffusion_model
from .sampling import sample_ddpm, sample_from_model
from .eval import compute_metrics, create_plots, evaluate_model, compute_kl_divergence
from .util import (
    sinusoidal_time_embedding, 
    compute_mmd_rbf, 
    precompute_diffusion_params
)
