# SAE-Probes Reproduction Findings
*Updated: August 12, 2025*

## Executive Summary

**Current Status**: After correcting data alignment issues and ensuring fair comparisons between SAE and baseline methods, we find that SAE-based probes generally underperform baseline probes across most settings. The notable exception is out-of-distribution (OOD) tasks, where SAE probes show substantial improvements.

**Key Finding**: SAE probes underperform baselines in:
- Scarcity settings: -0.0101 mean AUC
- Class imbalance: -0.0155 mean AUC  
- Overall: -0.0126 mean AUC across 4,403 matched conditions

However, SAE probes excel at OOD generalization with +0.1072 mean AUC improvement.

**Confidence Note**: While our methodology is sound, we have limited confidence in the broader conclusions pending additional validation and critique. The OOD overperformance finding is particularly interesting but requires further investigation.

## Paper Context

**Paper**: "Are Sparse Autoencoders Useful?" (https://arxiv.org/pdf/2502.16681)

The paper examines whether SAE features provide advantages for probing tasks compared to traditional baseline methods. Our reproduction attempt has revealed complex findings:

1. **Alignment with paper**: SAE probes generally do not outperform baselines (consistent with paper's skepticism)
2. **Key discrepancy**: Our OOD results show SAE superiority, while the paper reports baseline superiority for covariate-shifted data
3. **Interpretation challenge**: The specific types of distribution shifts tested may differ between implementations

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

### 1. PRIMARY FINDING: OOD Overperformance by SAE Probes

**The Notable Exception**: While SAE probes underperform in most settings, they show remarkable superiority for out-of-distribution generalization:

**Performance Results**:
- **Success rate**: 100% (all 8 OOD datasets show SAE improvement)
- **Mean improvement**: +0.107 AUC (substantial and consistent)
- **Best case**: `90_glue_qnli` (+0.266 AUC: 0.644 → 0.911)
- **Worst case**: `87_glue_cola` (+0.042 AUC: 0.791 → 0.833)

**Important Caveat - Contradicts Paper**:
- **Paper reports**: "baselines outperform SAE probes when generalizing to covariate shifted data"
- **Our results**: Complete opposite - SAEs consistently outperform baselines on OOD
- **Implication**: This discrepancy requires careful investigation before drawing strong conclusions

**Possible Explanations for OOD Success**:
1. **Semantic features**: SAE features may capture more transferable semantic patterns
2. **Robustness**: Less reliance on dataset-specific statistical regularities
3. **Feature selection**: Our top-k selection process may be particularly effective for OOD
4. **Implementation differences**: Our OOD definition may differ from the paper's linguistic shifts

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

## Confidence Assessment and Limitations

### High Confidence Findings
- **Methodology correction**: The condition matching problem was real and significant
- **General underperformance**: SAE probes consistently underperform in scarcity and class imbalance settings
- **Implementation quality**: Core probe training and evaluation methodology is sound

### Moderate Confidence Findings  
- **OOD overperformance**: While consistent in our results, it contradicts the paper and needs validation
- **Feature selection impact**: Our top-k selection may contribute to OOD success
- **Relative performance magnitudes**: The specific AUC differences observed

### Low Confidence Areas
- **Generalizability**: Limited to Gemma-2-9b model and specific datasets
- **OOD interpretation**: Uncertain if our OOD tasks match the paper's intended distribution shifts
- **Normal setting conclusions**: Only 1 dataset tested (0.9% coverage)

### Critical Uncertainties
1. **OOD discrepancy explanation**: Why do our results contradict the paper so dramatically?
2. **Implementation differences**: How much do subtle choices affect conclusions?
3. **Broader applicability**: Would findings hold for other models, SAE architectures, or tasks?

## Recommendations

1. **Always verify condition matching** when comparing methods across different experimental settings
2. **Be skeptical of aggregate statistics** that combine different experimental conditions
3. **SAEs should be used selectively**:
   - ✅ Good for: OOD generalization, transfer learning (pending validation)
   - ❌ Poor for: Limited data, class imbalance, standard classification
4. **Traditional methods remain strong baselines** and often outperform complex approaches
5. **Require additional validation** before drawing strong conclusions about OOD performance

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

Our reproduction effort has yielded mixed but informative results:

### Core Finding
**SAE-based probes generally underperform baseline methods** when experimental conditions are properly matched:
- Scarcity: -0.0101 mean AUC (32.2% improvement rate)
- Class imbalance: -0.0155 mean AUC (14.7% improvement rate)
- Overall: -0.0126 mean AUC across 4,403 matched conditions

This aligns with the paper's skepticism about SAE utility for probing tasks.

### Notable Exception
**Out-of-distribution generalization shows dramatic SAE superiority** (+0.107 mean AUC, 100% success rate). This is our most interesting finding, though it directly contradicts the paper's reported results. This discrepancy warrants significant further investigation.

### Current Assessment
While we don't have complete confidence in all conclusions, the evidence suggests:
1. **Traditional methods are more robust** for standard probing tasks, especially with limited data
2. **SAE features may excel at capturing transferable patterns** that generalize across distributions
3. **The value of SAEs is highly context-dependent** - not a universal improvement

### Next Steps Required
1. **Validate OOD findings** with additional experiments and different distribution shift types
2. **Investigate discrepancy** between our OOD results and the paper's findings
3. **Expand coverage** to more models, datasets, and SAE architectures
4. **Peer review and critique** of methodology and conclusions

The most valuable outcome of this reproduction is not just the specific results, but the discovery of how sensitive conclusions can be to implementation details and the importance of fair experimental comparisons.

---

## Data Sources and Visualizations

### Primary Result Datasets
All analyses are based on probe performance results stored in CSV format:

**Baseline Probe Results** (`results/baseline_probes_gemma-2-9b/`):
- Normal setting: `normal_settings/layer20_results.csv`
- Scarcity setting: `scarcity/all_results.csv` (2,247 conditions)
- Class imbalance: `imbalance/all_results.csv` (2,147 conditions)
- OOD setting: `ood/all_results.csv` (8 datasets)

**SAE Probe Results** (`results/sae_probes_gemma-2-9b/`):
- Normal setting: `normal_setting/all_metrics.csv` (1 dataset only)
- Scarcity setting: `scarcity_setting/all_metrics.csv` (113 datasets)
- Class imbalance: `class_imbalance_setting/all_metrics.csv` (113 datasets)
- OOD setting: `OOD_setting/all_metrics.csv` (8 datasets)

### Key Visualizations

**Matched Comparison Visualizations** (`figures/matched_comparison/`):
- `sae_vs_baseline_matched_all.png` - Main comparison with properly matched conditions
- `scarcity_analysis.png` - Detailed scarcity performance by training size
- `class_imbalance_analysis.png` - Performance across different imbalance ratios
- Generated by: `visualize_sae_vs_baseline_matched.py`

**Initial Comparison Visualizations** (`figures/key_comparison/`):
- `sae_vs_baseline_all_settings.png` - Overview across all settings (unmatched)
- `sae_vs_baseline_scarcity.png` - Scarcity setting detail
- `sae_vs_baseline_class_imbalance.png` - Class imbalance detail
- `sae_vs_baseline_ood.png` - OOD setting detail
- `performance_distribution.png` - Distribution of performance differences
- Generated by: `visualize_sae_vs_baseline.py` (Note: uses unmatched comparisons)

**OOD Analysis** (`figures/ood_analysis/`):
- `ood_sae_vs_baseline.png` - Detailed OOD performance comparison
- `ood_distribution.png` - Distribution of OOD improvements
- Shows consistent SAE superiority across all 8 OOD datasets

### Analysis Scripts
- `visualize_sae_vs_baseline_matched.py` - Corrected analysis with proper condition matching
- `visualize_sae_vs_baseline.py` - Original analysis (flawed matching)
- `create_results_summary.py` - Generates summary statistics
- Scripts located in repository root directory

**Key Files**:
- Corrected analysis: `visualize_sae_vs_baseline_matched.py`
- Findings documents: 
  - `michael_documentation/MATCHED_COMPARISON_FINDINGS.md`
  - `michael_documentation/OOD_INVESTIGATION.md`
  - `michael_documentation/CRITICAL_UPDATE_SUMMARY.md`
- Visualizations: `figures/matched_comparison/` and `figures/key_comparison/`
- Raw results: `results/baseline_probes_gemma-2-9b/` and `results/sae_probes_gemma-2-9b/`