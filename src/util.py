import math
import numpy as np
import torch
from typing import Tuple

def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int = 32) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half)
    args = timesteps[..., None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
    return emb


def compute_mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma=None) -> float:
    n = X.shape[0]
    m = Y.shape[0]
    XX = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    YY = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    XY = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    
    if sigma is None:
        # median heuristic
        dists = np.concatenate([XX.flatten(), YY.flatten(), XY.flatten()])
        sigma = np.sqrt(0.5 * np.median(dists[dists > 0])) + 1e-6
    
    Kxx = np.exp(-XX / (2 * sigma ** 2))
    Kyy = np.exp(-YY / (2 * sigma ** 2))
    Kxy = np.exp(-XY / (2 * sigma ** 2))
    
    return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())


def precompute_diffusion_params(timesteps: int, beta_start: float, beta_end: float, device: torch.device):
    betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
    alphas = 1.0 - betas
    alpha_cum = torch.cumprod(alphas, dim=0)  # \bar{alpha}_t
    sqrt_alpha_cum = torch.sqrt(alpha_cum)
    sqrt_one_minus_alpha_cum = torch.sqrt(1 - alpha_cum)
    
    # for discrete posterior variance (from DDPM paper)
    alphas_prev = torch.cat([torch.tensor([1.0], device=device), alpha_cum[:-1]], dim=0)
    posterior_variance = betas * (1. - alphas_prev) / (1. - alpha_cum)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alpha_cum': alpha_cum,
        'sqrt_alpha_cum': sqrt_alpha_cum,
        'sqrt_one_minus_alpha_cum': sqrt_one_minus_alpha_cum,
        'posterior_variance': posterior_variance,
        'alphas_prev': alphas_prev
    }
