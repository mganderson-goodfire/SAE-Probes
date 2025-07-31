#!/usr/bin/env python3
"""
OOD Comparison: The critical test of SAE interpretability
If SAE features are truly more semantic/interpretable, they should 
generalize better to out-of-distribution tasks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures/ood_analysis', exist_ok=True)

print("=== OOD SETTING: THE KEY TEST OF SAE INTERPRETABILITY ===\n")

# Load OOD results
baseline_ood = pd.read_csv('results/baseline_probes_gemma-2-9b/ood/all_results.csv')
sae_ood = pd.read_csv('results/sae_probes_gemma-2-9b/OOD_setting/all_metrics.csv')

print(f"Baseline OOD: {len(baseline_ood)} records on {baseline_ood['dataset'].nunique()} datasets")
print(f"SAE OOD: {len(sae_ood)} records on {sae_ood['dataset'].nunique()} datasets")

# Get unique datasets
baseline_datasets = set(baseline_ood['dataset'].unique())
sae_datasets = set(sae_ood['dataset'].unique())
common_datasets = baseline_datasets.intersection(sae_datasets)

print(f"\nCommon datasets: {common_datasets}")

# For each dataset, get best baseline and best SAE
comparisons = []
for dataset in common_datasets:
    # Best baseline
    baseline_subset = baseline_ood[baseline_ood['dataset'] == dataset]
    if len(baseline_subset) == 0:
        continue
    # Handle different column names in OOD baseline
    auc_col = 'test_auc' if 'test_auc' in baseline_subset.columns else 'test_auc_baseline'
    best_baseline_idx = baseline_subset[auc_col].idxmax()
    best_baseline = baseline_subset.loc[best_baseline_idx]
    
    # Best SAE
    sae_subset = sae_ood[sae_ood['dataset'] == dataset]
    if len(sae_subset) == 0:
        continue
    best_sae_idx = sae_subset['test_auc'].idxmax()
    best_sae = sae_subset.loc[best_sae_idx]
    
    comparisons.append({
        'dataset': dataset,
        'baseline_auc': best_baseline[auc_col],
        'baseline_method': best_baseline.get('method', 'unknown'),
        'sae_auc': best_sae['test_auc'],
        'sae_k': best_sae.get('k', 'unknown'),
        'improvement': best_sae['test_auc'] - best_baseline[auc_col]
    })

comparison_df = pd.DataFrame(comparisons)
comparison_df = comparison_df.sort_values('improvement', ascending=False)

print("\n=== OOD RESULTS ===")
print(comparison_df.to_string(index=False))

# Create main visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left plot: Direct comparison scatter
ax1.scatter(comparison_df['baseline_auc'], comparison_df['sae_auc'], 
           s=100, alpha=0.7, edgecolor='black', linewidth=1)

# Add diagonal line
min_val = min(comparison_df['baseline_auc'].min(), comparison_df['sae_auc'].min()) - 0.05
max_val = max(comparison_df['baseline_auc'].max(), comparison_df['sae_auc'].max()) + 0.05
ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No improvement')

# Add dataset labels
for _, row in comparison_df.iterrows():
    ax1.annotate(row['dataset'].split('_')[-1], 
                (row['baseline_auc'], row['sae_auc']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax1.set_xlabel('Best Baseline AUC', fontsize=12)
ax1.set_ylabel('Best SAE Probe AUC', fontsize=12)
ax1.set_title('OOD: SAE vs Baseline Performance', fontsize=14)
ax1.grid(alpha=0.3)
ax1.set_xlim(min_val, max_val)
ax1.set_ylim(min_val, max_val)

# Calculate statistics
mean_improvement = comparison_df['improvement'].mean()
pct_improved = (comparison_df['improvement'] > 0).sum() / len(comparison_df) * 100
pct_worse = (comparison_df['improvement'] < 0).sum() / len(comparison_df) * 100

ax1.text(0.05, 0.95, f'Mean improvement: {mean_improvement:.3f}\n'
                     f'Datasets improved: {pct_improved:.0f}%\n'
                     f'Datasets worse: {pct_worse:.0f}%',
         transform=ax1.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Right plot: Improvement by dataset
datasets = comparison_df['dataset'].values
improvements = comparison_df['improvement'].values
colors = ['green' if imp > 0 else 'red' for imp in improvements]

bars = ax2.bar(range(len(datasets)), improvements, color=colors, alpha=0.7, edgecolor='black')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_xticks(range(len(datasets)))
ax2.set_xticklabels([d.replace('_', ' ') for d in datasets], rotation=45, ha='right')
ax2.set_ylabel('AUC Improvement (SAE - Baseline)', fontsize=12)
ax2.set_title('OOD Performance Improvement by Dataset', fontsize=14)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, imp in zip(bars, improvements):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{imp:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

plt.tight_layout()
plt.savefig('figures/ood_analysis/ood_sae_vs_baseline.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/ood_analysis/ood_sae_vs_baseline.pdf', bbox_inches='tight')
print(f"\nSaved: figures/ood_analysis/ood_sae_vs_baseline.png/pdf")

# Additional analysis: Performance distribution
fig, ax = plt.subplots(figsize=(10, 6))

# Create box plots
data_to_plot = [comparison_df['baseline_auc'].values, comparison_df['sae_auc'].values]
positions = [1, 2]
bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))

# Add individual points
for i, (baseline, sae) in enumerate(zip(comparison_df['baseline_auc'], comparison_df['sae_auc'])):
    ax.plot([1, 2], [baseline, sae], 'o-', alpha=0.5, color='gray', markersize=6)

ax.set_xticks([1, 2])
ax.set_xticklabels(['Best Baseline', 'Best SAE'])
ax.set_ylabel('Test AUC', fontsize=12)
ax.set_title('OOD Performance Distribution: Baseline vs SAE', fontsize=14)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/ood_analysis/ood_distribution.png', dpi=150, bbox_inches='tight')
print(f"Saved: figures/ood_analysis/ood_distribution.png")

# Print key insights
print("\n" + "="*60)
print("KEY INSIGHTS FROM OOD ANALYSIS")
print("="*60)
print(f"1. Overall improvement: {mean_improvement:.3f} AUC")
print(f"2. Success rate: {pct_improved:.0f}% of datasets improved with SAE")
print(f"3. Failure rate: {pct_worse:.0f}% of datasets got worse with SAE")
print(f"4. Best improvement: {comparison_df.iloc[0]['dataset']} (+{comparison_df.iloc[0]['improvement']:.3f})")
print(f"5. Worst performance: {comparison_df.iloc[-1]['dataset']} ({comparison_df.iloc[-1]['improvement']:.3f})")

# Honest conclusion based on results
if mean_improvement > 0.01:
    print("\nCONCLUSION: SAE features show meaningful generalization advantage on OOD tasks")
    print("This supports the hypothesis that SAE features capture more transferable concepts.")
elif mean_improvement > 0.001:
    print("\nCONCLUSION: SAE features show marginal generalization advantage on OOD tasks")
    print("The improvement is too small to strongly support the interpretability hypothesis.")
elif mean_improvement > -0.001:
    print("\nCONCLUSION: SAE features perform comparably to baseline methods on OOD tasks")
    print("No evidence that SAE features provide better generalization.")
else:
    print("\nCONCLUSION: SAE features UNDERPERFORM baseline methods on OOD tasks")
    print(f"Average degradation: {abs(mean_improvement):.3f} AUC")
    print("This challenges the claim that SAE features are more interpretable/transferable.")
    
# Additional messaging for mixed results
if pct_improved > 0 and pct_worse > 0:
    print(f"\nMIXED RESULTS: SAEs help {pct_improved:.0f}% of tasks but hurt {pct_worse:.0f}%")
    print("Performance is task-dependent, suggesting SAE features may not be universally better.")

print("="*60)