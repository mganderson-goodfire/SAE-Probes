# SAE-Probes Reproduction Findings
*Updated: August 5, 2025*

## Executive Summary

**Critical Update**: Discovered major data alignment issues that invalidated initial findings. When properly matching experimental conditions between SAE and baseline methods, SAE probes actually perform WORSE than traditional methods in most settings, with only out-of-distribution (OOD) tasks showing genuine SAE benefits.

**Paper Context**: The original paper "Are Sparse Autoencoders Useful?" concludes that SAEs show "similar results with simple non-SAE baselines" and questions their utility. Our corrected analysis actually provides stronger evidence AGAINST SAE utility than the paper itself suggests.

## Major Discovery: Condition Matching Problem

### The Issue
The visualization code was comparing SAE and baseline methods without ensuring they used the same experimental conditions:
- **Scarcity**: Compared SAEs trained on large datasets (e.g., 1024 samples) vs baselines trained on small datasets (e.g., 10 samples)
- **Class Imbalance**: Didn't match the class imbalance ratios between methods
- This made SAE probes appear much better than they actually are

### Impact on Results

| Setting | Unmatched Comparison | Properly Matched | True Performance |
|---------|---------------------|------------------|------------------|
| Overall | 78.0% improved (+0.0758 AUC) | 23.8% improved (-0.0126 AUC) | SAEs WORSE |
| Scarcity | 81.4% improved (+0.0800 AUC) | 32.2% improved (-0.0101 AUC) | SAEs WORSE |
| Class Imbalance | 8.0% improved (-0.0100 AUC) | 14.7% improved (-0.0155 AUC) | SAEs WORSE |
| OOD | 100% improved (+0.1072 AUC) | 100% improved (+0.1072 AUC) | SAEs BETTER |

## Experimental Setup

- **Model**: Gemma-2-9b
- **Layer**: 20 (primary analysis layer)
- **Hardware**: M3 MacBook Pro (Apple Silicon)
- **Environment**: Python 3.11 with uv package management
- **Initial Analysis**: July 28-29, 2025
- **Corrected Analysis**: August 5, 2025

## Data Coverage

### Baseline Probes ✅ Complete
- **Normal**: 113 datasets × 5 methods = 565 results
- **Scarcity**: 2,247 conditions (dataset × training_size combinations)
- **Class Imbalance**: 2,147 conditions (dataset × imbalance_ratio combinations)
- **OOD**: 8 datasets

### SAE Probes ⚠️ Partially Complete
- **Normal**: Only 1 dataset (`5_hist_fig_ismale`) - 0.9% coverage
- **Scarcity**: 113 datasets complete - 100% coverage when matched
- **Class Imbalance**: 113 datasets complete - 100% coverage when matched
- **OOD**: 8 datasets complete - 100% coverage

## Corrected Key Findings

### 1. OOD Setting - Surprising Contradiction with Paper

**Our Performance Results**:
- All 8 datasets showed improvement with SAE probes
- Mean improvement: +0.107 AUC (substantial)
- Best: `90_glue_qnli` (+0.266 AUC: 0.644 → 0.911)
- Worst: `87_glue_cola` (+0.042 AUC: 0.791 → 0.833)

**Critical Discrepancy with Paper**:
- **Paper reports**: "baselines outperform SAE probes when generalizing to covariate shifted data"
- **Paper's OOD types**: Language changes, syntactic alterations, character substitutions
- **Our results**: Complete opposite - SAEs consistently outperform baselines

**Possible Explanations**:
1. **Different OOD definitions**: Paper uses linguistic shifts; ours may differ
2. **SAE selection bias**: We test multiple SAEs and pick best; paper uses single SAE
3. **Feature selection**: Our top-k selection may favor SAEs
4. **Implementation differences**: Train/test split or data processing variations

### 2. Scarcity Setting - SAEs Struggle with Limited Data

**Performance by Training Size**:
- 2-5 samples: SAEs perform -0.02 to -0.06 AUC worse
- 10-50 samples: SAEs perform ~-0.01 AUC worse
- 100+ samples: SAEs approach baseline performance
- Overall: 66.9% of conditions show SAEs performing worse

**Key Insight**: Traditional methods (especially logistic regression) are more sample-efficient than SAE-based approaches.

### 3. Class Imbalance - Consistent SAE Underperformance

**Performance**:
- Mean change: -0.0155 AUC (worst setting for SAEs)
- Only 14.7% of conditions show improvement
- 72.4% of conditions show SAEs performing worse
- Consistent underperformance across all imbalance ratios (5% to 45% minority class)

### 4. Normal Setting - Insufficient Data

**Limitation**: Only 1 dataset tested (`5_hist_fig_ismale`)
- Baseline best: 0.9940 AUC (Logistic Regression)
- SAE best: 0.9952 AUC
- Improvement: +0.0012 AUC (marginal)
- **Cannot generalize from single datapoint**

## Technical Analysis

### Why SAEs Underperform in Most Settings

1. **Feature Sparsity**: SAE features are extremely sparse, making it harder to learn robust classifiers with limited data
2. **Feature Selection Overhead**: The k-feature selection process adds complexity without proportional benefit
3. **Regularization Differences**: L1 regularization for SAEs vs L2 for baselines may not be optimal
4. **Dimensionality**: SAEs expand the feature space (e.g., 3584 → 131k dimensions) requiring more data

### Why SAEs Excel at OOD

1. **Semantic Features**: SAE features may capture more semantic/interpretable patterns
2. **Transferability**: These semantic features generalize better to distribution shifts
3. **Robustness**: Less reliance on dataset-specific patterns

## Methodology Validation

### ✅ Correctly Implemented
- Core probe training methodology
- Cross-validation and hyperparameter search
- Performance metrics calculation

### ❌ Critical Flaw Discovered
- Visualization code didn't match experimental conditions
- This created an unfair comparison favoring SAEs
- Once corrected, the true performance emerged

### ✅ Fix Implemented
- Created `visualize_sae_vs_baseline_matched.py` with proper condition matching
- Validates that comparisons use same (dataset, condition) pairs
- Generates both aggregate and per-setting visualizations

## Implications for the Paper's Conclusions

### Original Paper Claims
"SAEs show similar results with simple non-SAE baselines"

### Our Findings
SAEs show **worse** results than baselines in most settings:
- Scarcity: -0.0101 mean AUC
- Class Imbalance: -0.0155 mean AUC
- Overall: -0.0126 mean AUC across 4,403 matched conditions

### Exception: OOD Generalization
The +0.107 AUC improvement in OOD settings is substantial and suggests SAE features do have value for transfer learning tasks.

## Recommendations

1. **Always verify condition matching** when comparing methods across different experimental settings
2. **Be skeptical of aggregate statistics** that combine different experimental conditions
3. **SAEs should be used selectively**:
   - ✅ Good for: OOD generalization, transfer learning
   - ❌ Poor for: Limited data, class imbalance, standard classification
4. **Traditional methods remain strong baselines** and often outperform complex approaches

## Reproduction Quality Assessment

### ✅ What We Successfully Reproduced
1. Core experimental methodology
2. Probe training for both baselines and SAEs
3. Multiple experimental settings
4. Discovered and fixed critical comparison flaw

### ⚠️ Limitations
1. Normal setting has only 1 dataset (not representative)
2. Missing label_noise setting entirely
3. Limited to single model (Gemma-2-9b)
4. Hardware constraints prevented 1M width SAE testing

## Conclusion

Our reproduction reveals that **SAE probes generally underperform traditional methods** when experimental conditions are properly matched. The only exception in our results is out-of-distribution generalization, where SAE features show substantial advantages (+0.107 AUC).

However, this OOD finding **directly contradicts the paper**, which reports that baselines outperform SAEs on covariate-shifted data. This discrepancy suggests:

1. **OOD performance is highly dependent on the type of distribution shift**
2. **Our OOD implementation may differ significantly from the paper's**
3. **SAE selection methodology matters** - choosing best of multiple SAEs vs using a single SAE

The broader finding remains consistent: SAE probes are not universally better than traditional methods and often perform worse, especially with limited data or class imbalance.

The key lessons: 
- **Always ensure fair comparisons by matching experimental conditions**
- **Be cautious about generalizing OOD results** - performance may vary greatly by shift type
- **Implementation details matter** - seemingly minor differences can reverse conclusions

---

**Key Files**:
- Corrected analysis: `visualize_sae_vs_baseline_matched.py`
- Findings document: `MATCHED_COMPARISON_FINDINGS.md`
- Visualizations: `figures/matched_comparison/`