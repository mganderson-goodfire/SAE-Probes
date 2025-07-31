#!/usr/bin/env python3
"""
Compare SAE vs baseline performance for the dataset we actually tested
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create figures directory
os.makedirs('figures/comparison', exist_ok=True)

# Load data
baseline = pd.read_csv('results/baseline_probes_gemma-2-9b/normal_settings/layer20_results.csv')
sae_normal = pd.read_csv('results/sae_probes_gemma-2-9b/normal_setting/all_metrics.csv')

# Get the dataset we tested
test_dataset = '5_hist_fig_ismale'

# Get baseline results for this dataset
baseline_subset = baseline[baseline['dataset'] == test_dataset]

if len(baseline_subset) > 0:
    print(f"Found baseline results for {test_dataset}")
    print(f"Baseline methods tested: {list(baseline_subset['method'].values)}")
    print(f"Baseline AUCs: {list(baseline_subset['test_auc'].values)}")
    
    # Get best baseline
    best_baseline_idx = baseline_subset['test_auc'].idxmax()
    best_baseline = baseline_subset.loc[best_baseline_idx]
    print(f"\nBest baseline: {best_baseline['method']} with AUC {best_baseline['test_auc']:.4f}")
    
    # Get best SAE result
    best_sae_idx = sae_normal['test_auc'].idxmax()
    best_sae = sae_normal.loc[best_sae_idx]
    print(f"Best SAE: {best_sae['sae_id']} with k={best_sae['k']} and AUC {best_sae['test_auc']:.4f}")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: All methods comparison
    all_methods = list(baseline_subset['method'].values) + ['Best SAE']
    all_aucs = list(baseline_subset['test_auc'].values) + [best_sae['test_auc']]
    
    x_pos = np.arange(len(all_methods))
    bars = ax1.bar(x_pos, all_aucs, alpha=0.7)
    
    # Color the best performer
    best_idx = np.argmax(all_aucs)
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(1.0)
    
    # Color baseline methods blue, SAE method orange
    for i in range(len(baseline_subset)):
        bars[i].set_color('steelblue')
    if len(all_methods) > len(baseline_subset):
        bars[-1].set_color('darkorange')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(all_methods, rotation=45, ha='right')
    ax1.set_ylabel('Test AUC')
    ax1.set_title(f'All Methods Comparison for {test_dataset}')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0.9, 1.0)  # Zoom in on high performance region
    
    # Add value labels
    for i, v in enumerate(all_aucs):
        ax1.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: SAE performance across k values
    # Group by k and get mean performance
    k_performance = sae_normal.groupby('k')['test_auc'].agg(['mean', 'std', 'count']).reset_index()
    
    ax2.errorbar(k_performance['k'], k_performance['mean'], 
                 yerr=k_performance['std'], 
                 marker='o', capsize=5, linewidth=2, markersize=8)
    
    # Add horizontal line for best baseline
    ax2.axhline(y=best_baseline['test_auc'], color='steelblue', linestyle='--', 
                label=f'Best baseline ({best_baseline["method"]}): {best_baseline["test_auc"]:.4f}')
    
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Number of SAE features (k)')
    ax2.set_ylabel('Mean Test AUC (across SAE configs)')
    ax2.set_title('SAE Performance vs k')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0.98, 1.0)  # Zoom in
    
    plt.tight_layout()
    plt.savefig('figures/comparison/sae_vs_baseline_detailed.png', dpi=150)
    print("\nSaved: figures/comparison/sae_vs_baseline_detailed.png")
    
    # Create summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Dataset: {test_dataset}")
    print(f"\nBaseline methods:")
    for _, row in baseline_subset.iterrows():
        print(f"  - {row['method']}: {row['test_auc']:.4f}")
    
    print(f"\nSAE configurations tested: {sae_normal['sae_id'].nunique()}")
    print(f"k values tested: {sorted(sae_normal['k'].unique())}")
    print(f"\nSAE performance range: {sae_normal['test_auc'].min():.4f} - {sae_normal['test_auc'].max():.4f}")
    print(f"Best baseline AUC: {best_baseline['test_auc']:.4f}")
    print(f"Best SAE AUC: {best_sae['test_auc']:.4f}")
    print(f"Improvement: {best_sae['test_auc'] - best_baseline['test_auc']:.4f}")
    
    # Check how many SAE configs beat the best baseline
    sae_beats_baseline = (sae_normal['test_auc'] > best_baseline['test_auc']).sum()
    print(f"\nSAE configs that beat best baseline: {sae_beats_baseline}/{len(sae_normal)} ({sae_beats_baseline/len(sae_normal)*100:.1f}%)")
    
else:
    print(f"No baseline results found for dataset {test_dataset}")
    print("This might mean the dataset name format is different between baseline and SAE results.")