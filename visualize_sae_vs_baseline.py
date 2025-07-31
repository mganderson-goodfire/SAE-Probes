#!/usr/bin/env python3
"""
Direct comparison: SAE probes vs best baseline method
This answers: Are SAE probes better than the best traditional method?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures/key_comparison', exist_ok=True)

print("Loading baseline results...")
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

comparisons = []

for setting in ['normal', 'scarcity', 'class_imbalance', 'ood']:
    print(f"\nProcessing {setting} setting...")
    
    # Load baseline
    try:
        baseline_df = pd.read_csv(baseline_files[setting])
        print(f"  Baseline: {len(baseline_df)} records")
        print(f"  Baseline columns: {list(baseline_df.columns)}")
    except:
        print(f"  No baseline data for {setting}")
        continue
        
    # Load SAE
    try:
        sae_df = pd.read_csv(sae_files[setting])
        print(f"  SAE: {len(sae_df)} records")
    except:
        print(f"  No SAE data for {setting}")
        continue
    
    # Get best baseline per dataset
    if 'dataset' in baseline_df.columns and 'test_auc' in baseline_df.columns:
        # For settings with multiple conditions (scarcity, class_imbalance), group appropriately
        if setting == 'scarcity' and 'num_train' in baseline_df.columns:
            # Group by dataset AND training size
            best_baseline = baseline_df.loc[baseline_df.groupby(['dataset', 'num_train'])['test_auc'].idxmax()]
        elif setting == 'class_imbalance' and 'frac' in baseline_df.columns:
            # Group by dataset AND imbalance fraction
            best_baseline = baseline_df.loc[baseline_df.groupby(['dataset', 'frac'])['test_auc'].idxmax()]
        else:
            # Normal and OOD - just group by dataset
            best_baseline = baseline_df.loc[baseline_df.groupby('dataset')['test_auc'].idxmax()]
    else:
        print(f"  Missing required columns in baseline data")
        continue
        
    # Get best SAE per dataset
    if 'dataset' in sae_df.columns:
        best_sae = sae_df.loc[sae_df.groupby('dataset')['test_auc'].idxmax()]
    else:
        continue
    
    # Merge on dataset
    comparison = pd.merge(
        best_baseline[['dataset', 'test_auc', 'method']], 
        best_sae[['dataset', 'test_auc']], 
        on='dataset', 
        suffixes=('_baseline', '_sae')
    )
    comparison['setting'] = setting
    comparisons.append(comparison)
    print(f"  Matched datasets: {len(comparison)}")

if not comparisons:
    print("No valid comparisons found!")
    exit()

# Combine all comparisons
all_comparisons = pd.concat(comparisons, ignore_index=True)
print(f"\nTotal comparisons: {len(all_comparisons)}")

# Create the key visualization
plt.figure(figsize=(10, 10))

colors = {'normal': 'blue', 'scarcity': 'orange', 'class_imbalance': 'green', 'ood': 'red'}
markers = {'normal': 'o', 'scarcity': 's', 'class_imbalance': '^', 'ood': 'D'}

for setting in all_comparisons['setting'].unique():
    data = all_comparisons[all_comparisons['setting'] == setting]
    plt.scatter(data['test_auc_baseline'], data['test_auc_sae'], 
               c=colors.get(setting, 'gray'), 
               marker=markers.get(setting, 'o'),
               alpha=0.6, s=50, label=f'{setting} (n={len(data)})')

# Add diagonal line
min_val = min(all_comparisons['test_auc_baseline'].min(), all_comparisons['test_auc_sae'].min())
max_val = max(all_comparisons['test_auc_baseline'].max(), all_comparisons['test_auc_sae'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No improvement')

# Calculate overall statistics
improvements = all_comparisons['test_auc_sae'] - all_comparisons['test_auc_baseline']
mean_improvement = improvements.mean()
pct_improved = (improvements > 0).sum() / len(improvements) * 100

plt.xlabel('Best Baseline Method AUC', fontsize=12)
plt.ylabel('Best SAE Probe AUC', fontsize=12)
plt.title(f'SAE Probes vs Best Baseline Method Across All Settings\n'
          f'Mean improvement: {mean_improvement:.4f} | '
          f'{pct_improved:.1f}% improved | '
          f'n={len(all_comparisons)} datasets',
          fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

# Add marginal improvement line
plt.plot([min_val, max_val], [min_val + 0.01, max_val + 0.01], 'g:', alpha=0.5, label='+0.01 improvement')

plt.tight_layout()
plt.savefig('figures/key_comparison/sae_vs_baseline_all_settings.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/key_comparison/sae_vs_baseline_all_settings.pdf', bbox_inches='tight')
print("\nSaved: figures/key_comparison/sae_vs_baseline_all_settings.png/pdf")

# Print detailed statistics
print("\n" + "="*60)
print("DETAILED COMPARISON STATISTICS")
print("="*60)

for setting in all_comparisons['setting'].unique():
    data = all_comparisons[all_comparisons['setting'] == setting]
    setting_improvements = data['test_auc_sae'] - data['test_auc_baseline']
    
    print(f"\n{setting.upper()}:")
    print(f"  Datasets compared: {len(data)}")
    print(f"  Mean baseline AUC: {data['test_auc_baseline'].mean():.4f}")
    print(f"  Mean SAE AUC: {data['test_auc_sae'].mean():.4f}")
    print(f"  Mean improvement: {setting_improvements.mean():.4f}")
    print(f"  Datasets improved: {(setting_improvements > 0).sum()} ({(setting_improvements > 0).sum()/len(data)*100:.1f}%)")
    print(f"  Datasets worse: {(setting_improvements < 0).sum()} ({(setting_improvements < 0).sum()/len(data)*100:.1f}%)")

print("\n" + "="*60)
print(f"OVERALL: {pct_improved:.1f}% of datasets show improvement with SAE probes")
print(f"Mean improvement across all settings: {mean_improvement:.4f}")
print("="*60)