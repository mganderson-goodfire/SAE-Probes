# Critical Update: SAE vs Baseline Comparison Issues
*August 5, 2025*

## Executive Summary

We discovered a major flaw in the visualization code that made SAE probes appear much better than they actually are. When properly matching experimental conditions, SAE probes actually perform WORSE than traditional methods in most settings. 

Additionally, our OOD results directly contradict the paper's findings, suggesting significant implementation differences that require investigation.

## The Problem

The original visualization compared methods without ensuring they used the same experimental conditions:
- **Scarcity**: SAEs trained on 1024 samples were compared against baselines trained on 10 samples
- **Class Imbalance**: Different imbalance ratios were compared
- This created an unfair advantage for SAE methods

## The Fix

Created `visualize_sae_vs_baseline_matched.py` that properly matches conditions:
- Scarcity: Matches on (dataset, num_train)
- Class Imbalance: Matches on (dataset, frac)
- OOD/Normal: Matches on dataset only

## True Results

### Before (Unmatched)
- Overall: 78.0% improved, +0.0758 mean AUC
- Scarcity: 81.4% improved, +0.0800 mean AUC
- Class Imbalance: 8.0% improved, -0.0100 mean AUC
- OOD: 100% improved, +0.1072 mean AUC

### After (Properly Matched)
- Overall: 23.8% improved, -0.0126 mean AUC ❌
- Scarcity: 32.2% improved, -0.0101 mean AUC ❌
- Class Imbalance: 14.7% improved, -0.0155 mean AUC ❌
- OOD: 100% improved, +0.1072 mean AUC ✅

## Key Insights

1. **SAE probes excel at OOD tasks in our reproduction** - The +0.107 AUC improvement for out-of-distribution generalization is substantial and consistent. **However, this directly contradicts the paper's finding that baselines outperform SAEs on covariate-shifted data.**

2. **Traditional methods are more sample-efficient** - In scarcity settings, baselines significantly outperform SAEs, especially with very limited data (2-5 samples)

3. **SAEs struggle with class imbalance** - The worst setting for SAEs, with consistent underperformance across all imbalance ratios

4. **OOD results require careful interpretation** - The contradiction between our results and the paper suggests OOD performance is highly sensitive to implementation details and types of distribution shift

## Implications

- Always verify condition matching in comparative studies
- Be skeptical of aggregate statistics across different experimental conditions
- SAEs should be used selectively (good for OOD, poor for limited data)
- Traditional methods remain strong baselines

## Updated Documents

1. `REPRODUCTION_STATUS.md` - Updated with corrected results
2. `REPRODUCTION_FINDINGS.md` - Complete rewrite with new insights
3. `MATCHED_COMPARISON_FINDINGS.md` - Detailed analysis of the issue
4. `visualize_sae_vs_baseline_matched.py` - The corrected visualization code

## Visualizations

New properly matched visualizations available in:
- `figures/matched_comparison/sae_vs_baseline_matched_all.png`
- `figures/matched_comparison/scarcity_analysis.png`
- `figures/matched_comparison/class_imbalance_analysis.png`