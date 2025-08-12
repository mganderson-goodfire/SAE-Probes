# SAE-Probes Reproduction Status
*Updated: August 5, 2025*

## Summary
Successfully reproducing the paper "Are Sparse Autoencoders Useful?" on M3 MacBook Pro. **CRITICAL UPDATE: Discovered major data alignment issues in original analysis. When properly matching experimental conditions, SAE probes actually perform WORSE than baselines in most settings.**

**Key Finding**: Only the OOD (out-of-distribution) setting shows SAE benefits in our reproduction (+0.107 AUC, 100% success rate), but this **directly contradicts the paper's finding** that "baselines outperform SAE probes when generalizing to covariate shifted data." This discrepancy requires further investigation.

## Critical Discovery - August 5, 2025

### Condition Matching Issue
Discovered that the original visualization code was comparing SAE and baseline methods **without matching experimental conditions**:
- **Scarcity**: Compared SAEs trained on large datasets vs baselines trained on small datasets
- **Class Imbalance**: Didn't ensure same imbalance ratios were compared
- This made SAE probes appear much better than they actually are

### Corrected Results (Properly Matched Conditions)
| Setting | Original Result | Matched Result | True Performance |
|---------|----------------|----------------|------------------|
| Overall | 78.0% improved (+0.0758) | 23.8% improved (-0.0126) | SAEs WORSE |
| Scarcity | 81.4% improved (+0.0800) | 32.2% improved (-0.0101) | SAEs WORSE |
| Class Imbalance | 8.0% improved (-0.0100) | 14.7% improved (-0.0155) | SAEs WORSE |
| OOD | 100% improved (+0.1072) | 100% improved (+0.1072) | SAEs BETTER |
| Normal | 100% improved (+0.0012) | 100% improved (+0.0012) | Marginal (n=1) |

## Current Progress - August 5, 2025

### ✅ Completed
- **Model activations**: All available (extracted from tar files)
- **Baseline probe results**: Complete for all 5 experimental settings (pre-computed)
- **SAE probe training**: 
  - ✅ Normal: 70 results (1 dataset: `5_hist_fig_ismale`)
  - ✅ OOD: 32 results (8 datasets)
  - ✅ Scarcity: 8,988 results (113 datasets complete)
  - ✅ Class Imbalance: 8,588 results (113 datasets complete)
- **Visualizations**: Generated comprehensive comparison plots
- **Scripts fixed**: Memory-safe configuration, idempotent training

### 🔄 In Progress
- **Label noise setting**: Need SAE activation generation first

### ❌ Not Started
- **Label noise SAE activations**: Directory doesn't exist yet

## Experimental Matrix Status

| Setting         | SAE Probes           | Baseline Probes | SAE Activations   |
|-----------------|---------------------|-----------------|-------------------|
| normal          | ✅ 70 results       | ✅ Complete     | ✅ 15,614 files   |
| scarcity        | ✅ Complete (134 datasets) | ✅ Complete | ✅ 17,978 files   |
| class_imbalance | ✅ Complete (113 datasets) | ✅ Complete     | ✅ 17,178 files   |
| label_noise     | ❌ Not started      | ✅ Complete     | ❌ Missing        |
| OOD             | ✅ 32 results       | ✅ Complete     | ✅ 34 files       |

## Key Findings - Revised After Condition Matching

### 🎯 **OOD Results - The Only Clear SAE Success (BUT CONTRADICTS PAPER)**:
- **100% Success Rate**: All 8 datasets improved with SAE probes
- **Mean improvement**: +0.107 AUC (substantial improvement)
- **Best improvement**: `90_glue_qnli` (+0.266 AUC: 0.644 → 0.911)
- **CRITICAL DISCREPANCY**: The paper reports "baselines outperform SAE probes when generalizing to covariate shifted data"
- **Possible explanations**: Different OOD definitions, SAE selection methodology, or implementation differences

### ❌ **Scarcity Setting - SAEs Struggle with Limited Data**:
- **Success rate**: Only 32.2% of conditions improved
- **Mean change**: -0.0101 AUC (SAEs perform worse)
- **Worst with small datasets**: -0.02 to -0.06 AUC with 2-5 training samples
- **Key insight**: Traditional methods are more sample-efficient

### ❌ **Class Imbalance - SAEs Consistently Underperform**:
- **Success rate**: Only 14.7% of conditions improved
- **Mean change**: -0.0155 AUC (worst setting for SAEs)
- **Consistent across all imbalance ratios**: SAEs struggle regardless of class ratio

### ⚠️ **Normal Setting - Limited Evidence**:
- **Only 1 dataset tested**: Results not generalizable
- **Marginal improvement**: +0.0012 AUC
- **Coverage**: 0.9% of baseline conditions had matching SAE results

### Experimental Scale:
- **Total matched comparisons**: 4,403 properly matched conditions
- **Unmatched conditions excluded**: Thousands of invalid comparisons removed
- **Methods compared**: Best of 5 baseline approaches vs best SAE configuration per condition

## Next Actions (Priority Order)

### 1. Generate Label Noise Activations
```bash
# Check if script supports label_noise setting
python generate_sae_activations.py --model gemma-2-9b --setting label_noise
```

### 2. Train Label Noise SAE Probes
```bash
make train-sae-probes SETTING=label_noise MODEL=gemma-2-9b
# After generating activations
```

### 3. Final Analysis
```bash
make combine-results  # Aggregate all completed settings
make visualize        # Generate comprehensive plots
```

## Technical Details

### Memory Optimization
- **Fixed**: Removed 1M width SAEs from non-normal settings (line 242 in `train_sae_probes.py`)
- **Safe widths**: 16k and 131k work reliably on M3 MacBook Pro
- **Idempotent training**: Script skips existing files, safe to resume after interruption

### File Locations
- **SAE activations**: `data/sae_activations_gemma-2-9b/{setting}_setting/`
- **SAE probe models**: `data/sae_probes_gemma-2-9b/{setting}_setting/*.pkl`
- **Aggregated results**: `results/sae_probes_gemma-2-9b/{setting}_setting/all_metrics.csv`
- **Visualizations**: `figures/simple/`, `figures/detailed/`, `figures/comparison/`

### Latest Results Files
- Normal: `all_metrics_20250728_210729.csv` (timestamped)
- Scarcity: `all_metrics.csv` (8,988 records)
- Class Imbalance: `all_metrics.csv` (8,588 records)
- OOD: `all_metrics_20250728_210729.csv` (timestamped)

## Progress Timeline
- **July 22-24**: Initial SAE activation generation
- **July 28**: Fixed baseline training, generated initial visualizations  
- **July 29 AM**: Discovered scarcity training 51% complete, identified next steps
- **July 29 PM**: ✅ COMPLETED scarcity SAE probe training (113 datasets, 8,988 experiments)
- **July 29 PM**: ✅ COMPLETED class_imbalance SAE probe training (113 datasets, 8,588 experiments)
- **July 29 PM**: 🎯 **DISCOVERED COMPELLING OOD EVIDENCE** - 100% improvement rate, +0.107 mean AUC gain
- **July 29 PM**: ✅ COMPLETED comprehensive analysis and key visualizations

## Repository Health ✅
- All core scripts working and optimized
- Memory constraints identified and mitigated  
- Training process robust to interruption
- Comprehensive documentation and findings generated
- 4/5 experimental settings have SAE activations ready

Ready to complete the remaining SAE probe training and achieve full paper reproduction.