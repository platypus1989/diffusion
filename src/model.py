import torch
import torch.nn as nn
from diffusers import UNet1DModel
from .util import sinusoidal_time_embedding

class DiffusionMLP(nn.Module):
    def __init__(self, hidden=128, time_emb_dim=32, num_layers=2):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )
        
        # Build configurable MLP layers
        layers = []
        input_dim = 2 + time_emb_dim
        
        # Add hidden layers
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden, hidden),
                nn.ReLU(),
            ])
        
        # Add output layer
        layers.append(nn.Linear(hidden, 2))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        t_emb = sinusoidal_time_embedding(t, dim=32).to(x.dtype).to(x.device)
        t_emb = self.time_mlp(t_emb)
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)


def create_model(model_type="mlp", **kwargs):
    """Create diffusion model."""
    if model_type == "mlp":
        return DiffusionMLP(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
