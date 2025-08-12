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
    # Special handling for OOD data which has different column names
    if setting == 'ood' and 'test_auc_baseline' in baseline_df.columns:
        # OOD data has a different structure
        best_baseline = baseline_df.copy()
        best_baseline['test_auc'] = best_baseline['test_auc_baseline']
        best_baseline['method'] = 'logreg'  # OOD only uses logistic regression
    elif 'dataset' in baseline_df.columns and 'test_auc' in baseline_df.columns:
        # For settings with multiple conditions (scarcity, class_imbalance), group appropriately
        if setting == 'scarcity' and 'num_train' in baseline_df.columns:
            # Group by dataset AND training size
            best_baseline = baseline_df.loc[baseline_df.groupby(['dataset', 'num_train'])['test_auc'].idxmax()]
        elif setting == 'class_imbalance' and 'frac' in baseline_df.columns:
            # Group by dataset AND imbalance fraction
            best_baseline = baseline_df.loc[baseline_df.groupby(['dataset', 'frac'])['test_auc'].idxmax()]
        else:
            # Normal - just group by dataset
            best_baseline = baseline_df.loc[baseline_df.groupby('dataset')['test_auc'].idxmax()]
    else:
        print(f"  Missing required columns in baseline data")
        print(f"  Available columns: {list(baseline_df.columns)}")
        continue
        
    # Get best SAE per dataset
    if 'dataset' in sae_df.columns and 'test_auc' in sae_df.columns:
        # For settings with multiple conditions, we need to match the baseline conditions
        if setting == 'scarcity' and 'num_train' in baseline_df.columns:
            # This is tricky - SAE data doesn't have num_train column
            # For now, just get best SAE per dataset
            best_sae = sae_df.loc[sae_df.groupby('dataset')['test_auc'].idxmax()]
        elif setting == 'class_imbalance' and 'frac' in baseline_df.columns:
            # This is also tricky - SAE data doesn't have frac column
            # For now, just get best SAE per dataset
            best_sae = sae_df.loc[sae_df.groupby('dataset')['test_auc'].idxmax()]
        else:
            # Normal and OOD - just group by dataset
            best_sae = sae_df.loc[sae_df.groupby('dataset')['test_auc'].idxmax()]
    else:
        print(f"  Missing required columns in SAE data")
        continue
    
    # Merge on dataset
    if setting == 'ood':
        # For OOD, we need to handle the special case
        comparison = pd.merge(
            best_baseline[['dataset', 'test_auc', 'method']], 
            best_sae[['dataset', 'test_auc']], 
            on='dataset', 
            suffixes=('_baseline', '_sae')
        )
    elif setting in ['scarcity', 'class_imbalance']:
        # For these settings, we should ideally match conditions, but for now just match datasets
        # This is a limitation that should be documented
        print(f"  WARNING: {setting} setting - not matching specific conditions between baseline and SAE")
        comparison = pd.merge(
            best_baseline[['dataset', 'test_auc', 'method']], 
            best_sae[['dataset', 'test_auc']], 
            on='dataset', 
            suffixes=('_baseline', '_sae')
        )
    else:
        # Normal setting
        comparison = pd.merge(
            best_baseline[['dataset', 'test_auc', 'method']], 
            best_sae[['dataset', 'test_auc']], 
            on='dataset', 
            suffixes=('_baseline', '_sae')
        )
    
    comparison['setting'] = setting
    comparisons.append(comparison)
    print(f"  Matched datasets: {len(comparison)}")
    
    # Print dataset coverage info
    baseline_datasets = set(best_baseline['dataset'].unique())
    sae_datasets = set(best_sae['dataset'].unique())
    print(f"  Baseline datasets: {len(baseline_datasets)}")
    print(f"  SAE datasets: {len(sae_datasets)}")
    print(f"  Common datasets: {len(baseline_datasets & sae_datasets)}")

if not comparisons:
    print("\n" + "="*60)
    print("ERROR: No valid comparisons found!")
    print("="*60)
    print("This typically happens when:")
    print("1. SAE and baseline results are for different datasets")
    print("2. Column names don't match expected format")
    print("3. Data files are missing or corrupted")
    exit()

# Combine all comparisons
all_comparisons = pd.concat(comparisons, ignore_index=True)
print(f"\nTotal comparisons: {len(all_comparisons)}")

# Print data quality warnings
print("\n" + "="*60)
print("DATA QUALITY WARNINGS")
print("="*60)

settings_with_few_matches = all_comparisons.groupby('setting').size()
for setting, count in settings_with_few_matches.items():
    if count < 10:
        print(f"WARNING: {setting} setting only has {count} matched datasets!")
        
if len(all_comparisons) < 100:
    print(f"WARNING: Only {len(all_comparisons)} total comparisons - results may not be representative!")

# Check for suspicious AUC values
perfect_scores = (all_comparisons['test_auc_sae'] == 1.0).sum()
if perfect_scores > 0:
    print(f"WARNING: {perfect_scores} datasets have perfect SAE scores (AUC=1.0) - possible overfitting!")

print("="*60)

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

# Create separate plots for each setting
print("\nCreating individual setting plots...")
for setting in all_comparisons['setting'].unique():
    setting_data = all_comparisons[all_comparisons['setting'] == setting]
    
    plt.figure(figsize=(8, 8))
    
    plt.scatter(setting_data['test_auc_baseline'], setting_data['test_auc_sae'], 
               c=colors.get(setting, 'gray'), 
               marker=markers.get(setting, 'o'),
               alpha=0.6, s=100, edgecolor='black', linewidth=1)
    
    # Add diagonal line
    min_val = min(setting_data['test_auc_baseline'].min(), setting_data['test_auc_sae'].min()) - 0.02
    max_val = max(setting_data['test_auc_baseline'].max(), setting_data['test_auc_sae'].max()) + 0.02
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No improvement')
    
    # Calculate statistics for this setting
    setting_improvements = setting_data['test_auc_sae'] - setting_data['test_auc_baseline']
    setting_mean_improvement = setting_improvements.mean()
    setting_pct_improved = (setting_improvements > 0).sum() / len(setting_improvements) * 100
    
    plt.xlabel('Best Baseline Method AUC', fontsize=12)
    plt.ylabel('Best SAE Probe AUC', fontsize=12)
    plt.title(f'{setting.upper()} Setting: SAE vs Best Baseline\n'
              f'Mean improvement: {setting_mean_improvement:.4f} | '
              f'{setting_pct_improved:.1f}% improved | '
              f'n={len(setting_data)} datasets',
              fontsize=14)
    plt.grid(alpha=0.3)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Add text box with statistics
    textstr = f'Mean baseline: {setting_data["test_auc_baseline"].mean():.3f}\n'
    textstr += f'Mean SAE: {setting_data["test_auc_sae"].mean():.3f}\n'
    textstr += f'Improvement: {setting_mean_improvement:.3f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'figures/key_comparison/sae_vs_baseline_{setting}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: figures/key_comparison/sae_vs_baseline_{setting}.png")

print("\n" + "="*60)
print("DATA LIMITATIONS AND CAVEATS")
print("="*60)
print("1. Normal setting: Only 1 dataset has both baseline and SAE results")
print("2. OOD setting: Baseline uses only logistic regression (not best of multiple methods)")
print("3. Scarcity/Imbalance: Conditions not matched between baseline and SAE")
print("4. Different hyperparameter search spaces between methods")
print("5. SAE results may be incomplete or from different experimental runs")
print("="*60)