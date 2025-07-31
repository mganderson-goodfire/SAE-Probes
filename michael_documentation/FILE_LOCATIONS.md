# SAE-Probes File Locations Guide

This document describes where all generated files are saved in the SAE-Probes project.

## Model Activations (Pre-computed)
**Location**: `data/model_activations_{model_name}/`
- Format: `{dataset}_{layer_descriptor}.pt`
- Example: `100_news_fake_blocks.20.hook_resid_post.pt`
- These are extracted from tar files and contain the raw model hidden states

## SAE Activations (Generated)
**Base Location**: `data/sae_activations_{model_name}/`

### Normal Setting
- Path: `data/sae_activations_{model_name}/normal_setting/`
- Format: `{dataset}_{layer}_{width}_{l0}_X_train_sae.pt` (training activations)
- Format: `{dataset}_{layer}_{width}_{l0}_X_test_sae.pt` (test activations)
- Format: `{dataset}_{layer}_{width}_{l0}_y_train.pt` (training labels)
- Format: `{dataset}_{layer}_{width}_{l0}_y_test.pt` (test labels)
- Example: `100_news_fake_20_width_131k_average_l0_276_X_train_sae.pt`

### Scarcity Setting
- Path: `data/sae_activations_{model_name}/scarcity_setting/`
- Format: `{dataset}_{layer}_{width}_{l0}_{num_train}_X_train_sae.pt`
- Includes num_train in filename (10, 100, etc.)

### Class Imbalance Setting
- Path: `data/sae_activations_{model_name}/class_imbalance_setting/`
- Format: `{dataset}_{layer}_{width}_{l0}_frac{fraction}_X_train_sae.pt`
- Includes class balance fraction in filename

### OOD Setting
- Train path: `data/sae_activations_{model_name}/normal_setting/` (uses normal training data)
- Test path: `data/sae_activations_{model_name}/OOD_setting/` (OOD test data)

## Probe Results

### SAE Probe Results
**Location**: `data/sae_probes_{model_name}/{setting}_setting/`
- Format: `{dataset}_{layer}_{width}_{l0}_{reg_type}.pkl`
- Example: `5_hist_fig_ismale_20_width_131k_average_l0_276_l1.pkl`
- Contains: Dictionary with probe performance metrics (AUC, accuracy, etc.)

### Baseline Probe Results
**Location**: `data/baseline_results_{model_name}/{setting}/allruns/`
- Format: `layer{layer}_{dataset}_{method_name}.csv`
- Example: `layer20_100_news_fake_logreg.csv`
- Methods: logreg, knn3, knn5, knn10, xgboost, etc.

### Additional Baseline Outputs
- By dataset summaries: `data/baseline_results_{model_name}/{setting}/by_dataset/`
- Layer summaries: `results/baseline_probes_{model_name}/{setting}_settings/layer{layer}_results.csv`

## Combined Results (After Running combine_results.py)
**Location**: `results/sae_probes_{model_name}/{setting}_setting/`
- File: `all_metrics.csv`
- Contains: Aggregated probe performance data from all pickle files

## Key Variables in Filenames

### Model Names
- `gemma-2-9b`
- `llama-3.1-8b`
- `gemma-2-2b`

### Settings
- `normal` - Standard training/test split
- `scarcity` - Limited training data (10, 100 samples)
- `class_imbalance` - Imbalanced class ratios
- `label_noise` - Corrupted labels
- `OOD` - Out-of-distribution test data

### SAE Identifiers
- Width: `16k`, `131k`, `1m`
- L0: Average sparsity level (e.g., `average_l0_276`)
- Layer: Model layer number (e.g., `20`, `31`, `41`)

### Regularization Types
- `l1` - L1 regularization (sparse, used for SAE probes)
- `l2` - L2 regularization (dense, sometimes used for OOD)

## File Formats

### .pt Files (PyTorch tensors)
- SAE activations saved as sparse tensors to save space
- Can be loaded with `torch.load()` and converted to dense

### .pkl Files (Pickle)
- Probe results containing metrics dictionaries
- Loaded with `pickle.load()`

### .csv Files
- Baseline results and aggregated metrics
- Human-readable format for analysis

## Notes
- Files are created on-demand during processing
- Missing files are gracefully skipped in the updated scripts
- The optimized script processes all datasets per SAE to minimize loading overhead