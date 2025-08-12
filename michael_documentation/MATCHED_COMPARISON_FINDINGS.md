# Condition-Matched SAE vs Baseline Comparison: Key Findings

## Summary

When properly matching experimental conditions between SAE and baseline methods, the results tell a dramatically different story than the original unmatched comparison:

### Original (Unmatched) Results:
- **Overall**: 78.0% improved with mean +0.0758 AUC
- **Scarcity**: 81.4% improved with mean +0.0800 AUC
- **Class Imbalance**: 8.0% improved with mean -0.0100 AUC
- **OOD**: 100% improved with mean +0.1072 AUC

### Matched Comparison Results:
- **Overall**: 23.8% improved with mean -0.0126 AUC (SAEs perform WORSE overall!)
- **Scarcity**: 32.2% improved with mean -0.0101 AUC
- **Class Imbalance**: 14.7% improved with mean -0.0155 AUC
- **OOD**: 100% improved with mean +0.1072 AUC (unchanged)

## Key Insights

### 1. The Condition Matching Problem Was Severe
The original comparison was comparing SAE probes trained on larger datasets against baselines trained on smaller datasets in the scarcity setting. When we ensure both methods use the same training size, SAE probes actually perform worse in most cases.

### 2. SAE Probes Only Help in OOD Settings
The only setting where SAE probes consistently outperform baselines is out-of-distribution (OOD) generalization, with a substantial +0.107 AUC improvement. This suggests SAE features may capture more transferable representations.

### 3. SAE Probes Struggle with Limited Data
In the scarcity setting, SAE probes perform particularly poorly with very small training sets (2-5 samples), showing improvements of -0.02 to -0.06 AUC. They only approach baseline performance with larger training sets.

### 4. Class Imbalance is Challenging for SAEs
SAE probes consistently underperform baselines across all class imbalance ratios, with the gap being fairly consistent regardless of the imbalance level.

## Methodology Notes

### Matching Criteria:
- **Normal/OOD**: Matched on dataset only
- **Scarcity**: Matched on (dataset, num_train) pairs
- **Class Imbalance**: Matched on (dataset, frac) pairs with rounding to handle floating point

### Coverage:
- Normal: Only 0.9% coverage (1 dataset) - not representative
- Scarcity: 100% coverage (2247 conditions)
- Class Imbalance: 100% coverage (2147 conditions)  
- OOD: 100% coverage (8 datasets)

## Implications

1. **The original "suspicious" plot was indeed misleading** - it was comparing different experimental conditions, making SAE probes look much better than they actually are.

2. **SAE probes are not universally better** - they only show clear benefits for out-of-distribution generalization tasks.

3. **Traditional methods are more sample-efficient** - baseline methods handle limited data and class imbalance better than SAE-based approaches.

4. **The interpretability hypothesis needs refinement** - while SAE features may be more interpretable, this doesn't translate to better probe performance except in transfer/OOD scenarios.

## Recommendations

1. Always ensure experimental conditions are properly matched when comparing methods
2. Be cautious about claims that SAE features are universally superior for probing tasks
3. Consider the specific use case: SAEs may be preferred for OOD/transfer tasks but not for standard probing with limited data
4. Report results separately for each experimental condition rather than aggregating across different settings