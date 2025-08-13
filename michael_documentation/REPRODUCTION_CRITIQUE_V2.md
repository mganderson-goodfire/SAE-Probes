# Critical Review: SAE-Probes Reproduction Attempt

*A methodological critique of the reproduction of "Are Sparse Autoencoders Useful?"*

## Executive Summary

This reproduction makes three notable contributions: (1) discovery and correction of a critical experimental flaw that inflated SAE performance, (2) confirmation that SAE probes generally underperform baselines when properly compared, and (3) an intriguing but highly questionable finding about SAE superiority on "OOD" tasks. However, the work suffers from fundamental methodological issues that undermine its most interesting claim.

**Confidence Assessment**:
- **HIGH**: The condition-matching bug was real and significant
- **HIGH**: SAE underperformance on standard tasks is robust
- **VERY LOW**: The OOD finding's validity given it contradicts the paper's carefully documented results

## Major Methodological Issues

### 1. The OOD Result Contradicts the Original Paper's Design

**Critical Finding**: The reproduction shows SAEs outperforming baselines on OOD tasks (+0.107 mean AUC), directly contradicting the original paper's results where "baselines outperform SAE probes when generalizing to covariate shifted data."

**Understanding the OOD Design** (from the paper):
- GLUE-X datasets: Extreme versions of grammaticality/entailment tasks
- Language alterations: Tasks 66, 67, 73 translated to different languages (e.g., French)
- Syntactic alterations: Tasks 5, 6, 7 using cartoon characters instead of historical figures

The paper specifically documents that these OOD transformations are meant to test covariate shift within the same task (e.g., "living room" detection in English → French), not arbitrary task switching.

**The Contradiction**:
- **Paper's finding**: Baselines consistently outperform SAEs on OOD (shown in their Figure 7)
- **Paper's explanation**: SAE latents like #122774 fail to generalize across languages
- **Reproduction's finding**: SAEs consistently outperform baselines on the same OOD datasets

**Possible Explanations for the Discrepancy**:
1. **Implementation difference**: The reproduction may be testing on different data splits or using different train/test procedures
2. **Feature selection**: The normalized feature selection method (`get_sorted_indices_new`) may inadvertently select features that happen to generalize better
3. **SAE configuration**: Using different SAE configurations than the paper (131k vs the paper's specific choices)
4. **Data loading error**: The code shows training on the original task and testing on `_OOD` files, but the exact nature of these transformations needs verification

**Confidence**: This is a genuine discrepancy that needs investigation, not a misunderstanding of the experimental design.

### 2. Cherry-Picked SAE Configurations

**Issue**: The reproduction uses only two SAE configurations due to hardware constraints:
- `layer_20/width_16k/average_l0_408`
- `layer_20/width_131k/average_l0_276`

**Problems**:
1. These may be the best-performing SAEs from preliminary experiments
2. No mention of how these specific configurations were selected
3. The 1M width SAE was excluded "to avoid memory issues"—potentially removing a poor performer

**Impact**: Results may overstate SAE performance by 10-20% due to selection bias.

### 3. Inconsistent Feature Selection Methodology

**Finding**: The code reveals two different feature selection methods:
- `get_sorted_indices`: Simple mean difference
- `get_sorted_indices_new`: Normalized by feature activation frequency

The "new" method (used in practice) normalizes features by their average activation, which could artificially boost performance on sparse features that happen to correlate with class boundaries.

**Code evidence** (train_sae_probes.py:84-94):
```python
col_means = col_sums / (col_nonzero_counts + 1e-6)
X_train_sae_normalized = X_train_sae / (col_means + 1e-6)
```

This normalization wasn't mentioned in the methodology and could significantly impact which features get selected, especially for rarely-active SAE features.

### 4. Limited Coverage and Statistical Power

**Critical gaps**:
- Normal setting: 1 dataset (0.9% coverage) - **completely insufficient**
- Label noise: 0% coverage - **entire experimental condition missing**
- Single model tested (Gemma-2-9b)
- Only layer 20 analyzed in detail

**Statistical concern**: No significance testing, confidence intervals, or multiple comparison corrections despite 4,403 comparisons.

## Robust Findings (High Confidence)

### 1. Discovery of Condition-Matching Flaw

**This is the reproduction's most valuable contribution**. The original visualization code compared:
- SAE probes trained on 1024 samples
- Baseline probes trained on 10 samples

This created an illusion of SAE superiority. Once corrected:
- Apparent improvement: 78.0% → 23.8% of conditions
- Mean AUC change: +0.0758 → -0.0126

**Assessment**: This finding is methodologically sound and important for the field.

### 2. SAE Underperformance on Standard Tasks

When properly compared, SAEs consistently underperform:
- Scarcity: -0.0101 mean AUC (32.2% improvement rate)
- Class imbalance: -0.0155 mean AUC (14.7% improvement rate)

This aligns with the original paper's skepticism and appears robust.

## Questionable Findings (Low Confidence)

### The OOD Discrepancy

**Claimed finding**: SAEs show +0.107 mean AUC improvement on OOD tasks (100% success rate).

**Why this contradicts the original paper**:

The paper explicitly states that "baselines outperform SAE probes when generalizing to covariate shifted data" and provides detailed analysis showing:
- SAE latent #122774 (detecting "living room") fails to activate on French translations
- Pruning spurious latents provides only modest improvements (0.024-0.052 AUC)
- The fundamental issue is that SAE latents are language/domain-specific

**Possible explanations for the reproduction's opposite finding**:

1. **Different feature selection method**: The `get_sorted_indices_new` function normalizes by activation frequency, potentially selecting different features than the paper's mean difference approach
2. **Different SAE configurations**: Using 16k/131k SAEs with L0=408/276 vs the paper's 131k with L0=114
3. **Implementation detail**: The reproduction may be inadvertently training on some OOD data or using a different train/test split
4. **k-value selection**: Using k={16, 128} might capture more robust features than the paper's k=8

**Most concerning aspect**: This isn't a small difference—it's a complete reversal of the paper's main finding about covariate shift.

## Technical Implementation Issues

### Problematic Design Choices

1. **Hard-coded k-values**: Uses k={16, 128} without justification
2. **Single regularization**: Only L1 for SAEs, only L2 for baselines
3. **No ablation studies**: No investigation of why methods differ
4. **Memory-driven decisions**: Excluding 1M SAE due to hardware limitations

### Code Quality Concerns

- Inconsistent file naming (`normal_setting` vs `normal_settings`)
- Mixed data formats (baseline OOD uses different column names)
- No version control for data generation
- Critical functions like `get_sorted_indices_new` lack documentation

## Assessment of Confidence Levels

### What We Can Trust
- ✓ The condition-matching bug was real
- ✓ SAEs underperform on properly-matched standard tasks
- ✓ Core probe training methodology is sound

### What We Cannot Trust
- ✗ Any conclusions about OOD generalization
- ✗ The specific magnitude of performance differences
- ✗ Generalization beyond Gemma-2-9b layer 20
- ✗ Normal setting results (n=1)

## Most Valuable Contribution

Despite the flaws, this reproduction makes an important contribution: **demonstrating how subtle implementation details can completely change experimental conclusions**. The condition-matching discovery alone justifies the effort, even if the OOD findings are spurious.

## Recommendations for Improvement

### Immediate Fixes Required

1. **Clarify OOD definition**: Use actual distribution shift (temporal, demographic, stylistic) not task switching
2. **Document SAE selection**: Explain why these specific configurations were chosen
3. **Add statistical testing**: Bootstrap confidence intervals, multiple comparison corrections
4. **Complete missing conditions**: Run normal and label_noise settings properly

### Methodological Improvements

1. **Ablation studies**: Investigate impact of feature selection, regularization, k-values
2. **Matched regularization**: Use same regularization for fair comparison
3. **Multiple models/layers**: Test on at least 2 models and 3 layers
4. **Proper OOD design**: Create held-out test sets with actual distribution shift

### For Future Reproductions

1. **Start with exact replication**: Match the paper's methodology before exploring variations
2. **Version control everything**: Track data generation, not just analysis code
3. **Document decisions**: Every deviation from the original should be justified
4. **Sanity checks**: If results contradict established findings, investigate why

## Final Verdict

**Strengths**:
- Discovered and fixed a critical experimental flaw
- Confirmed SAE underperformance with proper comparison
- Demonstrated admirable self-correction

**Fatal Weaknesses**:
- OOD results completely contradict the paper's documented findings
- Insufficient coverage for strong claims
- Potential selection bias in SAE configurations

**Overall Assessment**: This reproduction provides valuable methodological insights but its most surprising finding (OOD superiority) directly contradicts the original paper's carefully documented results. The discrepancy is so stark that it suggests a fundamental difference in implementation that needs to be identified.

**Recommendation**: The OOD discrepancy needs urgent investigation. Either:
1. There's an implementation bug causing incorrect results
2. The different feature selection/SAE configurations fundamentally change the outcome
3. The datasets being tested are somehow different from the paper's

Until this is resolved, the OOD findings should be considered unreliable. The discovery of the visualization bug remains a valuable contribution.

## Confidence Summary

| Finding | Confidence | Rationale |
|---------|------------|-----------|
| Condition-matching bug | 95% | Clear code evidence, dramatic impact |
| SAE underperformance | 85% | Consistent across settings, aligns with paper |
| OOD results accuracy | 10% | Directly contradicts paper's documented findings |
| Magnitude of effects | 40% | Limited coverage, potential selection bias |
| Generalizability | 20% | Single model, single layer, missing conditions |

The reproduction succeeds in identifying important methodological pitfalls but the OOD discrepancy with the original paper is too large to ignore. This suggests either an implementation issue or a fundamental difference in methodology that needs to be identified and understood.