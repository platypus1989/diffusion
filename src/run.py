#!/usr/bin/env python3
"""
Main script for running diffusion model training and evaluation.
"""

import argparse
import json
import os
import random
import numpy as np
import torch
from typing import Dict, Any

# Fix imports to work from root directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import TwoMoonsDataset
from src.model import create_model
from src.train import train_diffusion_model
from src.sampling import sample_from_model
from src.eval import evaluate_model


def set_seed(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        # Data configuration
        'n_samples': 10000,
        'noise': 0.08,
        'random_state': 1,
        'normalize': True,
        'test_ratio': 0.2,
        
        # Model configuration
        'model_type': 'mlp', 
        'hidden': 128,
        'time_emb_dim': 32,
        'num_layers': 2,
        
        # Training configuration
        'batch_size': 256,
        'lr': 2e-4,
        'epochs': 60,
        'timesteps': 200,
        'beta_start': 1e-4,
        'beta_end': 0.02,
        'save_every': 10,
        
        # Sampling configuration
        'n_generated_samples': 2000,
        
        # Evaluation configuration
        'n_eval_samples': 2000,
        
        # System configuration
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_dir': 'outputs',
        'seed': 1
    }


def main():
    """Main function - trains a vanilla diffusion model from scratch and evaluates it."""
    parser = argparse.ArgumentParser(description='Train and evaluate diffusion models on 2D distributions (always trains from scratch)')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp'],
                       help='Type of model to use (only MLP is suitable for 2D point data)')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden layer size for MLP (default: 128)')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of hidden layers in MLP (default: 2)')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Test set size ratio (default: 0.2)')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    config['model_type'] = args.model
    config['hidden'] = args.hidden
    config['num_layers'] = args.num_layers
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    config['output_dir'] = args.output_dir if args.experiment_name is None else os.path.join(args.output_dir, args.experiment_name)
    config['seed'] = args.seed
    config['test_ratio'] = args.test_ratio
    
    # Set seed
    set_seed(config['seed'])
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config['output_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration: {config}")
    print(f"Device: {config['device']}")
    
    # Create dataset
    dataset = TwoMoonsDataset(
        n_samples=config['n_samples'],
        noise=config['noise'],
        random_state=config['seed'],
        normalize=config['normalize'],
        test_ratio=config['test_ratio']
    )
    
    # Get test data for evaluation
    test_data = dataset.test_data
    train_data = dataset.train_data
    mean, std = dataset.mean, dataset.std
    
    print(f"Dataset split:")
    print(f"  - Training samples: {len(train_data)}")
    print(f"  - Test samples: {len(test_data)}")
    print(f"  - Test ratio: {config['test_ratio']}")
    
    # Create model
    print(f"\nCreating {config['model_type']} model...")
    print(f"  - Hidden size: {config['hidden']}")
    print(f"  - Number of layers: {config['num_layers']}")
    print(f"  - Time embedding dim: {config['time_emb_dim']}")
    model = create_model(
        model_type=config['model_type'],
        hidden=config['hidden'],
        time_emb_dim=config['time_emb_dim'],
        num_layers=config['num_layers']
    )
    
    # Train model
    print("\nTraining model...")
    model, history, diffusion_params = train_diffusion_model(model, dataset, config)
    
    # Save training history
    history_path = os.path.join(config['output_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # Generate samples
    print(f"\nGenerating {config['n_generated_samples']} samples...")
    generated_samples = sample_from_model(model, config, config['n_generated_samples'])
    
    # Denormalize samples
    if mean is not None and std is not None:
        generated_samples = dataset.denormalize(generated_samples)
        test_data = dataset.denormalize(test_data)
        
    # Evaluate model
    print(f"\nEvaluating model...")
    results = evaluate_model(
        real_samples=test_data,
        generated_samples=generated_samples,
        n_samples=config['n_eval_samples'],
        output_dir=config['output_dir']
    )
    
    print(f"\nEvaluation complete! Results saved to {config['output_dir']}")
    print(f"MMD Score: {results['metrics']['mmd']:.6f}")
    print(f"KL Divergence: {results['metrics']['kl_divergence']:.6f}")


if __name__ == "__main__":
    main()
