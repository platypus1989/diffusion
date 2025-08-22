import torch
import numpy as np
from typing import Dict, Any
from .util import precompute_diffusion_params


@torch.no_grad()
def sample_ddpm(model, diffusion_params: Dict[str, torch.Tensor], 
                n_samples: int = 1000, device: torch.device = None) -> np.ndarray:
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    # Extract parameters
    betas = diffusion_params['betas']
    alphas = diffusion_params['alphas']
    alpha_cum = diffusion_params['alpha_cum']
    sqrt_alpha_cum = diffusion_params['sqrt_alpha_cum']
    sqrt_one_minus_alpha_cum = diffusion_params['sqrt_one_minus_alpha_cum']
    posterior_variance = diffusion_params['posterior_variance']
    alphas_prev = diffusion_params['alphas_prev']
    
    # Start from pure noise
    x = torch.randn(n_samples, 2, device=device)
    
    for t in reversed(range(len(betas))):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.float32)
        
        # Predict noise
        pred_eps = model(x, t_tensor)
        
        # Predict x0
        sqrt_ac = sqrt_alpha_cum[t]
        sqrt_1_ac = sqrt_one_minus_alpha_cum[t]
        pred_x0 = (x - sqrt_1_ac * pred_eps) / (sqrt_ac + 1e-8)
        
        # Posterior mean
        coef1 = betas[t] * torch.sqrt(alphas_prev[t]) / (1.0 - alpha_cum[t])
        coef2 = (1.0 - alphas_prev[t]) * torch.sqrt(alphas[t]) / (1.0 - alpha_cum[t])
        mu = coef1 * pred_x0 + coef2 * x
        
        # Add noise for t > 0
        if t > 0:
            var = posterior_variance[t]
            noise = torch.randn_like(x)
            x = mu + torch.sqrt(var) * noise
        else:
            x = mu
    
    return x.cpu().numpy()


def sample_from_model(model, config: Dict[str, Any], n_samples: int = 1000) -> np.ndarray:
    diffusion_params = precompute_diffusion_params(
        timesteps=config['timesteps'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
        device=config['device']
    )
    
    return sample_ddpm(model, diffusion_params, n_samples, config['device'])
