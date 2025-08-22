import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any
from .util import compute_mmd_rbf
from scipy.stats import entropy


def compute_kl_divergence(real_samples: np.ndarray, generated_samples: np.ndarray, 
                         bins: int = 50) -> float:
    """
    Compute KL divergence between real and generated samples using histogram estimation.
    
    Args:
        real_samples: Real data samples (n_samples, 2)
        generated_samples: Generated data samples (n_samples, 2)
        bins: Number of bins for histogram estimation
        
    Returns:
        KL divergence value (KL(P||Q) where P is real, Q is generated)
        Lower values indicate better distribution matching.
        Note: KL divergence is asymmetric and can be sensitive to binning.
    """
    # Create 2D histograms
    real_hist, x_edges, y_edges = np.histogram2d(
        real_samples[:, 0], real_samples[:, 1], 
        bins=bins, density=True
    )
    
    gen_hist, _, _ = np.histogram2d(
        generated_samples[:, 0], generated_samples[:, 1], 
        bins=bins, density=True
    )
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    real_hist = real_hist + epsilon
    gen_hist = gen_hist + epsilon
    
    # Normalize to get proper probability distributions
    real_hist = real_hist / np.sum(real_hist)
    gen_hist = gen_hist / np.sum(gen_hist)
    
    # Compute KL divergence: KL(P||Q) = sum(P * log(P/Q))
    # Only consider bins where real_hist > 0
    valid_mask = real_hist > 0
    kl_div = np.sum(real_hist[valid_mask] * np.log(real_hist[valid_mask] / gen_hist[valid_mask]))
    
    return float(kl_div)


def compute_metrics(real_samples: np.ndarray, generated_samples: np.ndarray,
                   n_samples: int = None) -> Dict[str, float]:
    # Subsample if needed
    if n_samples is not None:
        if len(real_samples) > n_samples:
            idx = np.random.choice(len(real_samples), n_samples, replace=False)
            real_sub = real_samples[idx]
        else:
            real_sub = real_samples
        
        if len(generated_samples) > n_samples:
            idx = np.random.choice(len(generated_samples), n_samples, replace=False)
            gen_sub = generated_samples[idx]
        else:
            gen_sub = generated_samples
    else:
        real_sub = real_samples
        gen_sub = generated_samples
    
    # Compute MMD
    mmd_score = compute_mmd_rbf(real_sub, gen_sub)
    
    # Compute KL divergence
    kl_divergence = compute_kl_divergence(real_sub, gen_sub)
    
    # Compute basic statistics
    real_mean = np.mean(real_sub, axis=0)
    real_std = np.std(real_sub, axis=0)
    gen_mean = np.mean(gen_sub, axis=0)
    gen_std = np.std(gen_sub, axis=0)
    
    # Mean and std differences
    mean_diff = np.linalg.norm(real_mean - gen_mean)
    std_diff = np.linalg.norm(real_std - gen_std)
    
    # Coverage metrics
    real_min, real_max = np.min(real_sub, axis=0), np.max(real_sub, axis=0)
    gen_min, gen_max = np.min(gen_sub, axis=0), np.max(gen_sub, axis=0)
    coverage_ratio = np.prod(np.minimum(gen_max - gen_min, real_max - real_min)) / np.prod(real_max - real_min)
    
    return {
        'mmd': mmd_score,
        'kl_divergence': kl_divergence,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'coverage_ratio': coverage_ratio,
        'n_real': len(real_sub),
        'n_generated': len(gen_sub)
    }


def create_plots(real_samples: np.ndarray, generated_samples: np.ndarray,
                n_samples: int = 2000, output_dir: str = "outputs") -> Dict[str, str]:
    """Create comparison plots."""
    # Subsample for plotting
    if len(real_samples) > n_samples:
        idx = np.random.choice(len(real_samples), n_samples, replace=False)
        real_sub = real_samples[idx]
    else:
        real_sub = real_samples
    
    if len(generated_samples) > n_samples:
        idx = np.random.choice(len(generated_samples), n_samples, replace=False)
        gen_sub = generated_samples[idx]
    else:
        gen_sub = generated_samples
    
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}
    
    # Side-by-side scatter plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Real Samples')
    plt.scatter(real_sub[:, 0], real_sub[:, 1], s=6, alpha=0.6, c='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.title('Generated Samples')
    plt.scatter(gen_sub[:, 0], gen_sub[:, 1], s=6, alpha=0.6, c='orange')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'scatter_comparison.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plot_paths['scatter'] = scatter_path
    plt.close()
    
    # Overlay plot
    plt.figure(figsize=(8, 8))
    plt.title('Real vs Generated Samples Overlay')
    plt.scatter(real_sub[:, 0], real_sub[:, 1], s=6, alpha=0.6, c='blue', label='Real')
    plt.scatter(gen_sub[:, 0], gen_sub[:, 1], s=6, alpha=0.6, c='orange', label='Generated')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    overlay_path = os.path.join(output_dir, 'overlay_comparison.png')
    plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
    plot_paths['overlay'] = overlay_path
    plt.close()
    
    return plot_paths


def evaluate_model(real_samples: np.ndarray, generated_samples: np.ndarray,
                  n_samples: int = 2000, output_dir: str = "outputs") -> Dict[str, Any]:
    print("Computing metrics...")
    metrics = compute_metrics(real_samples, generated_samples, n_samples)
    
    print("Creating visualizations...")
    plot_paths = create_plots(real_samples, generated_samples, n_samples, output_dir)
    
    # Save metrics
    import json
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    
    # Convert numpy types to native Python types for JSON serialization
    metrics_serializable = {}
    for k, v in metrics.items():
        if isinstance(v, np.integer):
            metrics_serializable[k] = int(v)
        elif isinstance(v, np.floating):
            metrics_serializable[k] = float(v)
        else:
            metrics_serializable[k] = v
    
    with open(results_path, 'w') as f:
        json.dump({
            'metrics': metrics_serializable,
            'plot_paths': plot_paths
        }, f, indent=2)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"MMD Score: {metrics['mmd']:.6f}")
    print(f"KL Divergence: {metrics['kl_divergence']:.6f}")
    print(f"Mean Difference: {metrics['mean_diff']:.6f}")
    print(f"Std Difference: {metrics['std_diff']:.6f}")
    print(f"Coverage Ratio: {metrics['coverage_ratio']:.6f}")
    
    return {
        'metrics': metrics,
        'plot_paths': plot_paths,
        'results_file': results_path
    }
