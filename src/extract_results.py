#!/usr/bin/env python3
"""
Extract experiment results from outputs directory and display MMD/KL divergence table.
"""

import json
import pandas as pd
from pathlib import Path


def main():
    layer_results = []
    hidden_size_results = []
    
    # Find all experiment directories in outputs
    for exp_dir in Path("outputs").iterdir():
        if not exp_dir.is_dir():
            continue
            
        eval_file = exp_dir / "evaluation_results.json"
        if not eval_file.exists():
            continue
            
        # Load results
        with open(eval_file, 'r') as f:
            data = json.load(f)
        
        metrics = data.get('metrics', {})
        
        # Determine experiment type
        if exp_dir.name.startswith('hidden_'):
            param = int(exp_dir.name.split('_')[1])
            hidden_size_results.append({
                'Hidden Size': param,
                'MMD': f"{metrics.get('mmd', 0):.6f}",
                'KL Divergence': f"{metrics.get('kl_divergence', 0):.6f}",
            })
        elif exp_dir.name.startswith('layer_'):
            param = int(exp_dir.name.split('_')[1])
            layer_results.append({
                'Number of Layers': param,
                'MMD': f"{metrics.get('mmd', 0):.6f}",
                'KL Divergence': f"{metrics.get('kl_divergence', 0):.6f}",
            })
        else:
            param = exp_dir.name
            
    
    # Create and display table
    df_hidden_size = pd.DataFrame(hidden_size_results).sort_values(by='Hidden Size', ascending=False)
    df_layer = pd.DataFrame(layer_results).sort_values(by='Number of Layers', ascending=False)
    print("# Hidden Size Results")
    print()
    print(df_hidden_size.to_markdown(index=False))
    print("# Number of Layers Results")
    print()
    print(df_layer.to_markdown(index=False))


if __name__ == "__main__":
    main()
