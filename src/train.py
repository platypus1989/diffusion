import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from typing import Dict, Any
from .util import precompute_diffusion_params


def train_diffusion_model(model, dataset, config: Dict[str, Any]):
    diffusion_params = precompute_diffusion_params(
        timesteps=config['timesteps'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
        device=config['device']
    )
    
    model = model.to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.MSELoss()
    dataloader = dataset.create_dataloader(
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
        split='train'
    )
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Training loop
    history = {'train_loss': []}
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (batch,) in enumerate(dataloader):
            batch = batch.to(config['device'])
            B = batch.shape[0]
            
            # Sample random timesteps and noise
            t = torch.randint(0, config['timesteps'], (B,), device=config['device'])
            eps = torch.randn_like(batch)
            
            # Forward diffusion
            sqrt_a = diffusion_params['sqrt_alpha_cum'][t].view(B, 1)
            sqrt_1_a = diffusion_params['sqrt_one_minus_alpha_cum'][t].view(B, 1)
            x_t = sqrt_a * batch + sqrt_1_a * eps
            
            # Predict and compute loss
            pred = model(x_t, t.float())
            loss = loss_fn(pred, eps)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        history['train_loss'].append(avg_loss)
        
        print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {avg_loss:.6f}")
            
    return model, history, diffusion_params
