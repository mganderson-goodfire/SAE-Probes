#!/usr/bin/env python3
"""
Detailed visualization of SAE probe results
Shows the multiple configurations tested
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create figures directory
os.makedirs('figures/detailed', exist_ok=True)

print("Loading SAE results...")
sae_normal = pd.read_csv('results/sae_probes_gemma-2-9b/normal_setting/all_metrics.csv')
sae_ood = pd.read_csv('results/sae_probes_gemma-2-9b/OOD_setting/all_metrics.csv')

print(f"\nNormal setting:")
print(f"- Total rows: {len(sae_normal)}")
print(f"- Unique datasets: {sae_normal['dataset'].nunique()}")
print(f"- Dataset: {sae_normal['dataset'].unique()}")
print(f"- Columns: {list(sae_normal.columns)}")

print(f"\nOOD setting:")
print(f"- Total rows: {len(sae_ood)}")
print(f"- Unique datasets: {sae_ood['dataset'].nunique()}")
print(f"- Datasets: {sorted(sae_ood['dataset'].unique())}")

# Analyze normal setting in detail
if 'sae_id' in sae_normal.columns:
    print(f"\nNormal setting SAE configurations:")
    print(f"- Unique SAE IDs: {sae_normal['sae_id'].nunique()}")
    print(f"- SAE IDs: {sae_normal['sae_id'].unique()}")

if 'k' in sae_normal.columns:
    print(f"- k values tested: {sorted(sae_normal['k'].unique())}")

# Plot 1: Performance across different k values for the single dataset
if 'k' in sae_normal.columns:
    plt.figure(figsize=(10, 6))
    
    # Group by SAE ID and k
    for sae_id in sae_normal['sae_id'].unique():
        sae_data = sae_normal[sae_normal['sae_id'] == sae_id]
        sae_data_sorted = sae_data.sort_values('k')
        
        # Extract SAE info from ID
        if 'width' in sae_id:
            width = sae_id.split('width_')[1].split('/')[0]
            l0 = sae_id.split('l0_')[1]
            label = f"Width {width}, L0 {l0}"
        else:
            label = sae_id
            
        plt.plot(sae_data_sorted['k'], sae_data_sorted['test_auc'], 
                marker='o', label=label, linewidth=2, markersize=8)
    
    plt.xscale('log', base=2)
    plt.xlabel('Number of SAE features (k)')
    plt.ylabel('Test AUC')
    plt.title(f'SAE Performance vs k for dataset: {sae_normal["dataset"].iloc[0]}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/detailed/sae_k_comparison_normal.png', dpi=150)
    print("\nSaved: figures/detailed/sae_k_comparison_normal.png")

# Plot 2: OOD results summary
if len(sae_ood) > 0:
    # Get performance by dataset
    ood_summary = sae_ood.groupby('dataset')['test_auc'].agg(['mean', 'std', 'count']).reset_index()
    ood_summary = ood_summary.sort_values('mean', ascending=False)
    
    plt.figure(figsize=(12, 8))
    x_pos = np.arange(len(ood_summary))
    
    bars = plt.bar(x_pos, ood_summary['mean'], yerr=ood_summary['std'], 
                    capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    
    # Color bars by performance
    for i, bar in enumerate(bars):
        if ood_summary.iloc[i]['mean'] > 0.9:
            bar.set_color('lightgreen')
        elif ood_summary.iloc[i]['mean'] > 0.8:
            bar.set_color('yellow')
        elif ood_summary.iloc[i]['mean'] > 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('lightcoral')
    
    plt.xticks(x_pos, ood_summary['dataset'], rotation=45, ha='right')
    plt.ylabel('Mean Test AUC')
    plt.title('OOD SAE Probe Performance by Dataset')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
    plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Fair performance')
    plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Good performance')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/detailed/ood_dataset_performance.png', dpi=150)
    print("Saved: figures/detailed/ood_dataset_performance.png")
    
    # Print statistics
    print(f"\nOOD Performance Summary:")
    print(f"- Mean AUC across all datasets: {ood_summary['mean'].mean():.3f}")
    print(f"- Best performing dataset: {ood_summary.iloc[0]['dataset']} (AUC: {ood_summary.iloc[0]['mean']:.3f})")
    print(f"- Worst performing dataset: {ood_summary.iloc[-1]['dataset']} (AUC: {ood_summary.iloc[-1]['mean']:.3f})")
    print(f"- Datasets with AUC > 0.9: {(ood_summary['mean'] > 0.9).sum()}")
    print(f"- Datasets with AUC > 0.8: {(ood_summary['mean'] > 0.8).sum()}")
    print(f"- Datasets with AUC > 0.7: {(ood_summary['mean'] > 0.7).sum()}")

# Plot 3: Compare SAE configurations (if multiple)
if 'sae_id' in sae_normal.columns and sae_normal['sae_id'].nunique() > 1:
    # Get best k for each SAE
    best_per_sae = sae_normal.loc[sae_normal.groupby('sae_id')['test_auc'].idxmax()]
    
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(best_per_sae))
    
    bars = plt.bar(x_pos, best_per_sae['test_auc'], alpha=0.7)
    
    # Add k values as labels
    for i, (idx, row) in enumerate(best_per_sae.iterrows()):
        plt.text(i, row['test_auc'] + 0.001, f"k={row['k']}", 
                ha='center', va='bottom', fontsize=8)
    
    labels = []
    for sae_id in best_per_sae['sae_id']:
        if 'width' in sae_id:
            width = sae_id.split('width_')[1].split('/')[0]
            l0 = sae_id.split('l0_')[1]
            labels.append(f"{width}\nL0={l0}")
        else:
            labels.append(sae_id)
    
    plt.xticks(x_pos, labels)
    plt.ylabel('Best Test AUC')
    plt.title('Best Performance by SAE Configuration')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/detailed/sae_config_comparison.png', dpi=150)
    print("Saved: figures/detailed/sae_config_comparison.png")

print("\nDetailed visualization complete!")