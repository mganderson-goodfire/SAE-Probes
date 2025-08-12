# OOD Results Discrepancy Investigation
*August 5, 2025*

## Summary

Our reproduction shows SAE probes dramatically outperforming baselines on OOD tasks (+0.107 mean AUC, 100% success rate), while the paper reports the opposite: "baselines outperform SAE probes when generalizing to covariate shifted data."

This document investigates potential causes of this discrepancy.

## The Contradiction

### Paper's OOD Findings
- **Result**: Baselines outperform SAE probes
- **OOD Types**: 
  - Language changes (e.g., English → French translations)
  - Syntactic alterations
  - Character substitutions (historical figures → cartoon characters)
- **Methodology**: Single SAE (width=131k, L0=114), trained on original data, tested on shifted data
- **Example**: English "living room" SAE latent doesn't activate on French translation

### Our OOD Findings
- **Result**: SAE probes outperform baselines
- **Mean improvement**: +0.107 AUC
- **Success rate**: 100% (8/8 datasets)
- **Best case**: +0.266 AUC on `90_glue_qnli`
- **Methodology**: Multiple SAEs tested, best selected per dataset

## Potential Explanations

### 1. Different OOD Definitions
**Paper's OOD**: Linguistic and syntactic shifts within the same conceptual task
- English → French translation
- Formal → informal language
- Real people → fictional characters

**Our OOD**: May involve different types of distribution shifts
- Need to examine: `get_OOD_datasets()` and `get_OOD_traintest()` functions
- Check if our OOD involves concept shifts rather than linguistic shifts

### 2. SAE Selection Methodology

**Paper**: 
- Uses single pre-selected SAE (width=131k, L0=114)
- No optimization for the OOD task
- May not be the best SAE for generalization

**Our Approach**:
- Tests multiple SAE configurations
- Selects best-performing SAE per dataset
- Potential for overfitting to test distribution

### 3. Feature Selection Differences

**Our Implementation**:
- Uses top-k feature selection based on class mean differences
- k values: {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
- Selection optimized on training data

**Paper**: 
- Unclear if they use feature selection
- May use all SAE features or different selection criteria

### 4. Train/Test Split Strategy

**Possible Difference**:
- How is the training data selected for OOD experiments?
- Are we training on data that's more similar to the OOD test set?
- Check: `get_OOD_traintest()` implementation

## Investigation Steps

### 1. Examine OOD Dataset Definitions
```python
# Need to check:
- What datasets are in get_OOD_datasets()?
- How do they differ from the training data?
- Are they linguistic shifts or conceptual shifts?
```

### 2. Review SAE Selection Process
```python
# Check if we're:
- Using validation performance to select SAEs
- Potentially leaking test information
- Comparing best-of-many vs single fixed choice
```

### 3. Analyze Feature Selection Impact
```python
# Test:
- Performance without feature selection
- Performance with fixed k across all datasets
- Impact of selection criterion
```

### 4. Verify Train/Test Data Sources
```python
# Confirm:
- Training data source for OOD experiments
- Test data characteristics
- Similarity between train and OOD test
```

## Implications

### If Our Implementation is Correct:
- SAE performance on OOD is highly task-dependent
- Some distribution shifts favor SAEs, others favor baselines
- The type of OOD shift matters more than we thought

### If Our Implementation Has Issues:
- SAE selection bias could be inflating results
- Feature selection might be overfitting
- Train/test contamination possible

### Either Way:
- OOD generalization claims need careful qualification
- Implementation details can reverse conclusions
- Both results could be "correct" for their specific setups

## Recommendations

1. **Immediate**: Examine our OOD dataset definitions and compare to paper's description
2. **Short-term**: Test with fixed SAE selection (no optimization) to match paper's approach
3. **Medium-term**: Implement paper's specific OOD types (language translation, etc.)
4. **Long-term**: Systematic study of which OOD types favor SAEs vs baselines

## Conclusion

The dramatic difference between our OOD results and the paper's suggests that "out-of-distribution" is not a monolithic concept. Different types of distribution shifts may fundamentally favor different approaches. Our implementation may be testing a different kind of generalization than the paper intended.

This discrepancy highlights the importance of:
- Precise definitions of distribution shift
- Detailed methodology documentation
- Robustness testing across multiple OOD types