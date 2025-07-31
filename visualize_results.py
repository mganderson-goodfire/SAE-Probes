#!/usr/bin/env python3
"""
Simple visualization script for SAE probe results
Works with the data we've actually generated
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create figures directory
os.makedirs('figures/simple', exist_ok=True)

# Set plot style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.figsize': (8, 6)
})

print("Loading data...")

# Load baseline results for normal setting
baseline_normal = pd.read_csv('results/baseline_probes_gemma-2-9b/normal_settings/layer20_results.csv')
print(f"Loaded {len(baseline_normal)} baseline results for normal setting")

# Load SAE results
try:
    sae_normal = pd.read_csv('results/sae_probes_gemma-2-9b/normal_setting/all_metrics.csv')
    print(f"Loaded {len(sae_normal)} SAE results for normal setting")
except:
    print("No SAE results for normal setting")
    sae_normal = None

try:
    sae_ood = pd.read_csv('results/sae_probes_gemma-2-9b/OOD_setting/all_metrics.csv')
    print(f"Loaded {len(sae_ood)} SAE results for OOD setting")
except:
    print("No SAE results for OOD setting")
    sae_ood = None

# Plot 1: Baseline methods comparison
print("\nCreating baseline methods comparison...")
methods = baseline_normal['method'].unique()
method_means = []
method_stds = []

for method in methods:
    method_data = baseline_normal[baseline_normal['method'] == method]['test_auc']
    method_means.append(method_data.mean())
    method_stds.append(method_data.std())

plt.figure(figsize=(10, 6))
x_pos = np.arange(len(methods))
plt.bar(x_pos, method_means, yerr=method_stds, capsize=5, alpha=0.7)
plt.xticks(x_pos, methods)
plt.ylabel('Test AUC')
plt.title('Baseline Probe Performance by Method (Layer 20)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/simple/baseline_methods_comparison.png', dpi=150)
print("Saved: figures/simple/baseline_methods_comparison.png")

# Plot 2: SAE vs Best Baseline comparison (if SAE data exists)
if sae_normal is not None:
    print("\nCreating SAE vs baseline comparison...")
    
    # Get best baseline result per dataset
    best_baseline = baseline_normal.loc[baseline_normal.groupby('dataset')['test_auc'].idxmax()]
    
    # Get SAE results - pick best k for each dataset
    if 'k' in sae_normal.columns:
        best_sae = sae_normal.loc[sae_normal.groupby('dataset')['test_auc'].idxmax()]
    else:
        best_sae = sae_normal.groupby('dataset')['test_auc'].max().reset_index()
    
    # Merge on dataset
    comparison = pd.merge(
        best_baseline[['dataset', 'test_auc', 'method']], 
        best_sae[['dataset', 'test_auc']], 
        on='dataset', 
        suffixes=('_baseline', '_sae')
    )
    
    # Calculate improvement
    comparison['improvement'] = comparison['test_auc_sae'] - comparison['test_auc_baseline']
    
    # Plot scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(comparison['test_auc_baseline'], comparison['test_auc_sae'], alpha=0.6, s=50)
    
    # Add diagonal line
    min_val = min(comparison['test_auc_baseline'].min(), comparison['test_auc_sae'].min())
    max_val = max(comparison['test_auc_baseline'].max(), comparison['test_auc_sae'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No improvement')
    
    # Add labels
    plt.xlabel('Best Baseline Test AUC')
    plt.ylabel('Best SAE Test AUC')
    plt.title(f'SAE vs Baseline Probe Performance\nMean improvement: {comparison["improvement"].mean():.3f} ± {comparison["improvement"].std():.3f}')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/simple/sae_vs_baseline_scatter.png', dpi=150)
    print("Saved: figures/simple/sae_vs_baseline_scatter.png")
    
    # Plot improvement distribution
    plt.figure(figsize=(8, 6))
    plt.hist(comparison['improvement'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', label='No improvement')
    plt.axvline(comparison['improvement'].mean(), color='green', linestyle='-', 
                label=f'Mean: {comparison["improvement"].mean():.3f}')
    plt.xlabel('Improvement (SAE - Baseline)')
    plt.ylabel('Count')
    plt.title('Distribution of SAE Performance Improvement')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/simple/improvement_distribution.png', dpi=150)
    print("Saved: figures/simple/improvement_distribution.png")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"- Datasets compared: {len(comparison)}")
    print(f"- Mean baseline AUC: {comparison['test_auc_baseline'].mean():.3f}")
    print(f"- Mean SAE AUC: {comparison['test_auc_sae'].mean():.3f}")
    print(f"- Mean improvement: {comparison['improvement'].mean():.3f}")
    print(f"- Datasets improved: {(comparison['improvement'] > 0).sum()} ({(comparison['improvement'] > 0).sum() / len(comparison) * 100:.1f}%)")
    print(f"- Datasets worse: {(comparison['improvement'] < 0).sum()} ({(comparison['improvement'] < 0).sum() / len(comparison) * 100:.1f}%)")

# Plot 3: SAE performance by k value (if available)
if sae_normal is not None and 'k' in sae_normal.columns:
    print("\nCreating SAE performance by k plot...")
    
    k_performance = sae_normal.groupby('k')['test_auc'].agg(['mean', 'std', 'count']).reset_index()
    k_performance['se'] = k_performance['std'] / np.sqrt(k_performance['count'])
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(k_performance['k'], k_performance['mean'], 
                 yerr=1.96 * k_performance['se'], 
                 marker='o', capsize=5, capthick=2)
    plt.xscale('log', base=2)
    plt.xlabel('Number of SAE features (k)')
    plt.ylabel('Mean Test AUC')
    plt.title('SAE Probe Performance vs Number of Features')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/simple/sae_performance_by_k.png', dpi=150)
    print("Saved: figures/simple/sae_performance_by_k.png")

print("\nVisualization complete! Check the figures/simple/ directory.")