#!/usr/bin/env python3
"""
Create a results summary folder for SAE vs Baseline comparison results.
Specifically for Gemma-2-9b model analysis.
"""

import os
import shutil
from datetime import datetime

def create_results_summary(output_dir=None):
    """Create a comprehensive results summary folder."""
    
    # Default to Desktop if no output directory specified
    if output_dir is None:
        output_dir = os.path.expanduser("~/Desktop/SAE_Probes_Gemma2_9B_Results")
    
    # Create main directory and subdirectories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "matched_comparisons"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "performance_by_setting"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "documentation"), exist_ok=True)
    
    print(f"Creating results summary in: {output_dir}")
    
    # Create main README
    readme_content = f"""# SAE Probes vs Baseline Methods: Gemma-2-9b Model Results
*Generated: {datetime.now().strftime('%B %d, %Y')}*

## Executive Summary

This folder contains the results from reproducing "Are Sparse Autoencoders Useful?" using the **Gemma-2-9b model at layer 20**. After correcting for experimental condition matching, we find that SAE probes generally perform WORSE than traditional baseline methods.

## Key Finding

**When experimental conditions are properly matched:**
- Overall: SAE probes show **-0.0126 mean AUC** (worse than baselines)
- Success rate: Only **23.8%** of conditions show improvement with SAEs
- Exception: Out-of-distribution (OOD) tasks show **+0.107 AUC** improvement

## Contents

### 1. matched_comparisons/
- `sae_vs_baseline_matched_all.png` - Main comparison plot with proper condition matching
- `sae_vs_baseline_*.png` - Individual plots for each setting
- `scarcity_analysis.png` - Performance vs training set size
- `class_imbalance_analysis.png` - Performance vs class balance

### 2. performance_by_setting/
Detailed breakdowns for each experimental setting

### 3. documentation/
- Key findings and methodology notes
- OOD investigation details
- Comparison with paper findings

## Results Summary

| Setting | Mean AUC Change | Success Rate | Key Insight |
|---------|----------------|--------------|-------------|
| Scarcity | -0.0101 | 32.2% | Traditional methods more sample-efficient |
| Class Imbalance | -0.0155 | 14.7% | SAEs struggle with imbalanced data |
| OOD | +0.1072 | 100% | SAEs excel at distribution shift* |
| Normal | +0.0012 | 100% | Marginal improvement (n=1) |

*Note: OOD result contradicts the paper's findings - see documentation for details

## Important Context

1. **Model**: All results are for Gemma-2-9b model at layer 20
2. **Condition Matching**: Results reflect proper experimental condition matching
3. **OOD Discrepancy**: Our OOD results contradict the paper (SAEs win vs lose)
4. **Coverage**: Normal setting has limited coverage (1 dataset only)

## Key Takeaway

SAE probes are not universally better than traditional methods. They underperform in most settings, with the notable exception of out-of-distribution generalization (though this finding contradicts the original paper).
"""
    
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme_content)
    
    # Copy plots if they exist
    plot_mappings = [
        # Matched comparisons
        ("figures/matched_comparison/sae_vs_baseline_matched_all.png", 
         "matched_comparisons/overall_comparison.png"),
        ("figures/matched_comparison/scarcity_analysis.png", 
         "matched_comparisons/scarcity_analysis.png"),
        ("figures/matched_comparison/class_imbalance_analysis.png", 
         "matched_comparisons/class_imbalance_analysis.png"),
        
        # Individual setting plots (from key_comparison folder)
        ("figures/key_comparison/sae_vs_baseline_normal.png", 
         "performance_by_setting/normal_setting.png"),
        ("figures/key_comparison/sae_vs_baseline_scarcity.png", 
         "performance_by_setting/scarcity_setting.png"),
        ("figures/key_comparison/sae_vs_baseline_class_imbalance.png", 
         "performance_by_setting/class_imbalance_setting.png"),
        ("figures/key_comparison/sae_vs_baseline_ood.png", 
         "performance_by_setting/ood_setting.png"),
        
        # Additional analysis plots
        ("figures/ood_analysis/ood_sae_vs_baseline.png",
         "performance_by_setting/ood_detailed_analysis.png"),
    ]
    
    for src, dst in plot_mappings:
        if os.path.exists(src):
            dst_path = os.path.join(output_dir, dst)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src, dst_path)
            print(f"  Copied: {dst}")
        else:
            print(f"  Warning: {src} not found")
    
    # Copy documentation
    doc_files = [
        ("michael_documentation/MATCHED_COMPARISON_FINDINGS.md", 
         "documentation/matched_comparison_findings.md"),
        ("michael_documentation/OOD_INVESTIGATION.md", 
         "documentation/ood_investigation.md"),
        ("michael_documentation/CRITICAL_UPDATE_SUMMARY.md", 
         "documentation/critical_update_summary.md"),
    ]
    
    for src, dst in doc_files:
        if os.path.exists(src):
            dst_path = os.path.join(output_dir, dst)
            shutil.copy2(src, dst_path)
            print(f"  Copied: {dst}")
    
    # Create a concise findings summary
    findings_summary = """# Key Findings Summary

## The Condition Matching Problem

The original visualization code compared SAE and baseline methods without ensuring they used the same experimental conditions:
- **Scarcity**: Compared SAEs trained on large datasets (e.g., 1024 samples) vs baselines on small datasets (e.g., 10 samples)
- **Class Imbalance**: Didn't match the class imbalance ratios

This made SAE probes appear much better than they actually are.

## Corrected Results

### Before (Unmatched Conditions)
- Overall: 78.0% improved, +0.0758 mean AUC
- Scarcity: 81.4% improved, +0.0800 mean AUC

### After (Matched Conditions)
- Overall: 23.8% improved, -0.0126 mean AUC
- Scarcity: 32.2% improved, -0.0101 mean AUC

## OOD Discrepancy

**Our Results**: SAEs excel at OOD (+0.107 AUC, 100% success)
**Paper Results**: Baselines outperform SAEs on covariate shift

Possible explanations:
1. Different types of distribution shift
2. SAE selection methodology differences
3. Feature selection approach
4. Implementation variations

## Conclusions

1. SAE probes generally underperform traditional methods
2. Traditional methods are more sample-efficient
3. OOD performance depends heavily on implementation details
4. Always ensure fair comparisons by matching conditions
"""
    
    with open(os.path.join(output_dir, "documentation", "key_findings_summary.md"), "w") as f:
        f.write(findings_summary)
    
    print(f"\nResults summary created successfully in: {output_dir}")
    print("\nTo view the results, open the README.md file in the created folder.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create SAE probe results summary folder")
    parser.add_argument("--output", "-o", type=str, 
                        help="Output directory (default: ~/Desktop/SAE_Probes_Gemma2_9B_Results)")
    
    args = parser.parse_args()
    
    create_results_summary(args.output)