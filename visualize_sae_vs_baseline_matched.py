#!/usr/bin/env python3
"""
Condition-Matched Comparison: SAE probes vs baseline methods
This script properly matches experimental conditions between SAE and baseline methods.
For scarcity: matches on (dataset, num_train)
For class_imbalance: matches on (dataset, frac)
For normal/OOD: matches on dataset only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures/matched_comparison', exist_ok=True)

print("="*60)
print("CONDITION-MATCHED SAE VS BASELINE COMPARISON")
print("="*60)

# Load baseline results
baseline_files = {
    'normal': 'results/baseline_probes_gemma-2-9b/normal_settings/layer20_results.csv',
    'scarcity': 'results/baseline_probes_gemma-2-9b/scarcity/all_results.csv',
    'class_imbalance': 'results/baseline_probes_gemma-2-9b/imbalance/all_results.csv',
    'ood': 'results/baseline_probes_gemma-2-9b/ood/all_results.csv'
}

# Load SAE results
sae_files = {
    'normal': 'results/sae_probes_gemma-2-9b/normal_setting/all_metrics.csv',
    'scarcity': 'results/sae_probes_gemma-2-9b/scarcity_setting/all_metrics.csv', 
    'class_imbalance': 'results/sae_probes_gemma-2-9b/class_imbalance_setting/all_metrics.csv',
    'ood': 'results/sae_probes_gemma-2-9b/OOD_setting/all_metrics.csv'
}

def create_matched_comparisons(setting, baseline_df, sae_df):
    """Create properly matched comparisons based on setting type."""
    comparisons = []
    
    if setting == 'normal':
        # Match on dataset only
        baseline_best = baseline_df.loc[baseline_df.groupby('dataset')['test_auc'].idxmax()]
        sae_best = sae_df.loc[sae_df.groupby('dataset')['test_auc'].idxmax()]
        
        # Merge on dataset
        comparison = pd.merge(
            baseline_best[['dataset', 'test_auc', 'method']], 
            sae_best[['dataset', 'test_auc', 'k']], 
            on='dataset', 
            suffixes=('_baseline', '_sae')
        )
        
    elif setting == 'scarcity':
        # Match on (dataset, num_train)
        baseline_best = baseline_df.loc[baseline_df.groupby(['dataset', 'num_train'])['test_auc'].idxmax()]
        sae_best = sae_df.loc[sae_df.groupby(['dataset', 'num_train'])['test_auc'].idxmax()]
        
        # Merge on dataset AND num_train
        comparison = pd.merge(
            baseline_best[['dataset', 'num_train', 'test_auc', 'method']], 
            sae_best[['dataset', 'num_train', 'test_auc', 'k']], 
            on=['dataset', 'num_train'], 
            suffixes=('_baseline', '_sae')
        )
        
    elif setting == 'class_imbalance':
        # Match on (dataset, frac) with rounding for floating point issues
        # Round to 2 decimal places to handle floating point precision
        baseline_df = baseline_df.copy()
        sae_df = sae_df.copy()
        baseline_df['frac_rounded'] = baseline_df['ratio'].round(2)
        sae_df['frac_rounded'] = sae_df['frac'].round(2)
        
        baseline_best = baseline_df.loc[baseline_df.groupby(['dataset', 'frac_rounded'])['test_auc'].idxmax()]
        sae_best = sae_df.loc[sae_df.groupby(['dataset', 'frac_rounded'])['test_auc'].idxmax()]
        
        # Merge on dataset AND frac_rounded
        comparison = pd.merge(
            baseline_best[['dataset', 'frac_rounded', 'test_auc', 'method']], 
            sae_best[['dataset', 'frac_rounded', 'test_auc', 'k']], 
            on=['dataset', 'frac_rounded'], 
            suffixes=('_baseline', '_sae')
        )
        # Rename frac_rounded back to frac for consistency
        comparison.rename(columns={'frac_rounded': 'frac'}, inplace=True)
        
    elif setting == 'ood':
        # Special handling for OOD data format
        baseline_df = baseline_df.copy()
        baseline_df['test_auc'] = baseline_df['test_auc_baseline']
        baseline_df['method'] = 'logreg'  # OOD only uses logistic regression
        
        # Match on dataset only
        sae_best = sae_df.loc[sae_df.groupby('dataset')['test_auc'].idxmax()]
        
        comparison = pd.merge(
            baseline_df[['dataset', 'test_auc', 'method']], 
            sae_best[['dataset', 'test_auc', 'k']], 
            on='dataset', 
            suffixes=('_baseline', '_sae')
        )
    
    comparison['setting'] = setting
    return comparison

# Process each setting
all_comparisons = []
condition_stats = {}

for setting in ['normal', 'scarcity', 'class_imbalance', 'ood']:
    print(f"\nProcessing {setting} setting...")
    
    # Load data
    try:
        baseline_df = pd.read_csv(baseline_files[setting])
        sae_df = pd.read_csv(sae_files[setting])
    except:
        print(f"  ERROR: Could not load data for {setting}")
        continue
    
    # Create matched comparisons
    comparison = create_matched_comparisons(setting, baseline_df, sae_df)
    
    # Calculate statistics
    if setting == 'scarcity':
        n_baseline_conditions = baseline_df.groupby(['dataset', 'num_train']).size().shape[0]
        n_sae_conditions = sae_df.groupby(['dataset', 'num_train']).size().shape[0]
    elif setting == 'class_imbalance':
        baseline_df['frac_rounded'] = baseline_df['ratio'].round(2)
        sae_df['frac_rounded'] = sae_df['frac'].round(2)
        n_baseline_conditions = baseline_df.groupby(['dataset', 'frac_rounded']).size().shape[0]
        n_sae_conditions = sae_df.groupby(['dataset', 'frac_rounded']).size().shape[0]
    else:
        n_baseline_conditions = baseline_df['dataset'].nunique()
        n_sae_conditions = sae_df['dataset'].nunique()
    
    n_matched = len(comparison)
    coverage = n_matched / n_baseline_conditions * 100 if n_baseline_conditions > 0 else 0
    
    condition_stats[setting] = {
        'n_baseline_conditions': n_baseline_conditions,
        'n_sae_conditions': n_sae_conditions,
        'n_matched': n_matched,
        'coverage': coverage
    }
    
    print(f"  Baseline conditions: {n_baseline_conditions}")
    print(f"  SAE conditions: {n_sae_conditions}")
    print(f"  Matched conditions: {n_matched}")
    print(f"  Coverage: {coverage:.1f}%")
    
    if len(comparison) > 0:
        all_comparisons.append(comparison)

# Combine all comparisons
if not all_comparisons:
    print("\nERROR: No valid matched comparisons found!")
    exit()

all_comparisons_df = pd.concat(all_comparisons, ignore_index=True)

print(f"\nTotal matched comparisons: {len(all_comparisons_df)}")

# Calculate overall statistics
improvements = all_comparisons_df['test_auc_sae'] - all_comparisons_df['test_auc_baseline']
mean_improvement = improvements.mean()
pct_improved = (improvements > 0).sum() / len(improvements) * 100

# Create main visualization
plt.figure(figsize=(12, 10))

colors = {'normal': 'blue', 'scarcity': 'orange', 'class_imbalance': 'green', 'ood': 'red'}
markers = {'normal': 'o', 'scarcity': 's', 'class_imbalance': '^', 'ood': 'D'}

for setting in all_comparisons_df['setting'].unique():
    data = all_comparisons_df[all_comparisons_df['setting'] == setting]
    plt.scatter(data['test_auc_baseline'], data['test_auc_sae'], 
               c=colors.get(setting, 'gray'), 
               marker=markers.get(setting, 'o'),
               alpha=0.6, s=50, label=f'{setting} (n={len(data)})')

# Add diagonal line
min_val = min(all_comparisons_df['test_auc_baseline'].min(), all_comparisons_df['test_auc_sae'].min())
max_val = max(all_comparisons_df['test_auc_baseline'].max(), all_comparisons_df['test_auc_sae'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No improvement')

plt.xlabel('Best Baseline Method AUC', fontsize=12)
plt.ylabel('Best SAE Probe AUC', fontsize=12)
plt.title(f'Condition-Matched: SAE Probes vs Best Baseline Method\n'
          f'Mean improvement: {mean_improvement:.4f} | '
          f'{pct_improved:.1f}% improved | '
          f'n={len(all_comparisons_df)} matched conditions',
          fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/matched_comparison/sae_vs_baseline_matched_all.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/matched_comparison/sae_vs_baseline_matched_all.pdf', bbox_inches='tight')
print("\nSaved: figures/matched_comparison/sae_vs_baseline_matched_all.png/pdf")

# Print detailed statistics
print("\n" + "="*60)
print("MATCHED COMPARISON STATISTICS BY SETTING")
print("="*60)

for setting in all_comparisons_df['setting'].unique():
    data = all_comparisons_df[all_comparisons_df['setting'] == setting]
    setting_improvements = data['test_auc_sae'] - data['test_auc_baseline']
    
    print(f"\n{setting.upper()}:")
    print(f"  Conditions compared: {len(data)}")
    print(f"  Coverage: {condition_stats[setting]['coverage']:.1f}%")
    print(f"  Mean baseline AUC: {data['test_auc_baseline'].mean():.4f}")
    print(f"  Mean SAE AUC: {data['test_auc_sae'].mean():.4f}")
    print(f"  Mean improvement: {setting_improvements.mean():.4f}")
    print(f"  Conditions improved: {(setting_improvements > 0).sum()} ({(setting_improvements > 0).sum()/len(data)*100:.1f}%)")
    print(f"  Conditions worse: {(setting_improvements < 0).sum()} ({(setting_improvements < 0).sum()/len(data)*100:.1f}%)")

# Create setting-specific visualizations for scarcity and class_imbalance
for setting in ['scarcity', 'class_imbalance']:
    setting_data = all_comparisons_df[all_comparisons_df['setting'] == setting]
    
    if len(setting_data) == 0:
        continue
        
    if setting == 'scarcity':
        # Create plot showing performance vs training size
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Group by training size
        train_sizes = sorted(setting_data['num_train'].unique())
        baseline_means = []
        sae_means = []
        
        for size in train_sizes:
            size_data = setting_data[setting_data['num_train'] == size]
            baseline_means.append(size_data['test_auc_baseline'].mean())
            sae_means.append(size_data['test_auc_sae'].mean())
        
        # Plot means by training size
        ax1.plot(train_sizes, baseline_means, 'o-', label='Baseline (mean)', color='blue', markersize=8)
        ax1.plot(train_sizes, sae_means, 's-', label='SAE (mean)', color='orange', markersize=8)
        ax1.set_xscale('log')
        ax1.set_xlabel('Training Set Size', fontsize=12)
        ax1.set_ylabel('Mean Test AUC', fontsize=12)
        ax1.set_title('Performance vs Training Size', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot improvements by training size
        improvements_by_size = []
        for size in train_sizes:
            size_data = setting_data[setting_data['num_train'] == size]
            size_improvements = size_data['test_auc_sae'] - size_data['test_auc_baseline']
            improvements_by_size.append(size_improvements.mean())
        
        ax2.bar(range(len(train_sizes)), improvements_by_size, color='green', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xticks(range(len(train_sizes)))
        ax2.set_xticklabels([str(s) for s in train_sizes], rotation=45)
        ax2.set_xlabel('Training Set Size', fontsize=12)
        ax2.set_ylabel('Mean AUC Improvement (SAE - Baseline)', fontsize=12)
        ax2.set_title('SAE Improvement by Training Size', fontsize=14)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figures/matched_comparison/scarcity_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved: figures/matched_comparison/scarcity_analysis.png")
        
    elif setting == 'class_imbalance':
        # Create plot showing performance vs class imbalance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Group by imbalance fraction
        fracs = sorted(setting_data['frac'].unique())
        baseline_means = []
        sae_means = []
        
        for frac in fracs:
            frac_data = setting_data[setting_data['frac'] == frac]
            baseline_means.append(frac_data['test_auc_baseline'].mean())
            sae_means.append(frac_data['test_auc_sae'].mean())
        
        # Plot means by fraction
        ax1.plot(fracs, baseline_means, 'o-', label='Baseline (mean)', color='blue', markersize=8)
        ax1.plot(fracs, sae_means, 's-', label='SAE (mean)', color='green', markersize=8)
        ax1.set_xlabel('Positive Class Fraction', fontsize=12)
        ax1.set_ylabel('Mean Test AUC', fontsize=12)
        ax1.set_title('Performance vs Class Imbalance', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot improvements by fraction
        improvements_by_frac = []
        for frac in fracs:
            frac_data = setting_data[setting_data['frac'] == frac]
            frac_improvements = frac_data['test_auc_sae'] - frac_data['test_auc_baseline']
            improvements_by_frac.append(frac_improvements.mean())
        
        colors_bar = ['green' if imp > 0 else 'red' for imp in improvements_by_frac]
        ax2.bar(range(len(fracs)), improvements_by_frac, color=colors_bar, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xticks(range(len(fracs)))
        ax2.set_xticklabels([f'{f:.2f}' for f in fracs], rotation=45)
        ax2.set_xlabel('Positive Class Fraction', fontsize=12)
        ax2.set_ylabel('Mean AUC Improvement (SAE - Baseline)', fontsize=12)
        ax2.set_title('SAE Improvement by Class Imbalance', fontsize=14)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figures/matched_comparison/class_imbalance_analysis.png', dpi=150, bbox_inches='tight')
        print(f"Saved: figures/matched_comparison/class_imbalance_analysis.png")

print("\n" + "="*60)
print(f"OVERALL MATCHED COMPARISON: {pct_improved:.1f}% of conditions show improvement with SAE probes")
print(f"Mean improvement across all matched conditions: {mean_improvement:.4f}")
print("="*60)

print("\nMETHODOLOGY:")
print("- Normal/OOD: Matched on dataset only")
print("- Scarcity: Matched on (dataset, num_train)")
print("- Class Imbalance: Matched on (dataset, frac) with rounding to 2 decimals")
print("- Each comparison uses the best baseline method vs best SAE configuration")
print("="*60)