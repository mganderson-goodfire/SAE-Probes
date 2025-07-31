# Reproduction Plan: "Are Sparse Autoencoders Useful?"

This plan outlines how to reproduce the findings from the paper, with specific consideration for running on an M3 MacBook Pro (Apple Silicon) versus GPU clusters.

## Overview

The reproduction workflow has three main stages:
1. **Activation Generation** (GPU-intensive)
2. **Probe Training** (CPU-friendly)
3. **Visualization & Analysis** (CPU-friendly)

## Hardware Requirements

### M3 MacBook Pro Capabilities  
- ✅ Can run: ALL steps including SAE activation generation (using MPS)
- ✅ Model activations: Pre-computed and available
- ✅ SAE activation generation: Works for 16k and 131k width SAEs
- ⚠️ Limited by: Memory for 1M width SAEs (need 64GB+ RAM)
- ✅ Probe training, results analysis, visualization: All work perfectly

### GPU Cluster Requirements (Optional)
- Only needed for: 1M width SAEs or faster processing
- Model activation generation already done (included in repo)

## Reproduction Steps

### Option 1: Full Reproduction on M3 MacBook Pro (Recommended)

#### Step 1: Environment Setup
```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or traditional pip
python -m venv probing
source probing/bin/activate
pip install transformer_lens sae_lens transformers datasets torch xgboost sae_bench scikit-learn natsort
```

#### Step 2: Extract Pre-computed Model Activations
```bash
# Model activations are already generated and included
cd data/
tar -xzf model_activations_gemma-2-9b.tar.gz
tar -xzf model_activations_gemma-2-9b_OOD.tar.gz
```

#### Step 3: Generate SAE Activations (Works on M3!)
```bash
# Use optimized script - 10-100x faster
make generate-sae-acts-optimized

# This will generate activations for:
# - 16k width SAEs ✅
# - 131k width SAEs ✅  
# - 1M width SAEs ❌ (skip due to memory)
```

#### Step 4: Train Probes for All Experimental Settings

**Critical for Paper Reproduction**: The paper evaluates 5 challenging experimental conditions. Both baseline and SAE probes must be trained for each setting to enable comparison.

```bash
# OPTION 1: Train everything at once
make train-probes  # Trains all baselines + all SAE probes

# OPTION 2: Train by category
make train-all-baselines     # All 5 baseline settings
make train-all-sae-probes    # All 5 SAE settings

# OPTION 3: Train individually by setting
make train-baselines SETTING=normal MODEL=gemma-2-9b
make train-baselines SETTING=scarcity MODEL=gemma-2-9b
make train-baselines SETTING=class_imbalance MODEL=gemma-2-9b
make train-baselines SETTING=label_noise MODEL=gemma-2-9b
make train-baselines SETTING=OOD MODEL=gemma-2-9b

# Then SAE probes for same settings...
make train-sae-probes SETTING=normal MODEL=gemma-2-9b
# ... etc
```

**The 5 Experimental Settings**:
1. **`normal`**: Standard training conditions (baseline comparison)
2. **`scarcity`**: Limited training data (10, 100 samples) 
3. **`class_imbalance`**: 90% samples from one class
4. **`label_noise`**: 20% corrupted training labels
5. **`OOD`**: Out-of-distribution test data (covariate shift)

**Paper's Central Question**: Do SAEs help in challenging conditions where baseline methods struggle?

#### Step 5: Combine Results
```bash
make combine-results
```

#### Step 6: Generate Visualizations
```bash
make visualize
# Or run individual notebooks
```

### Option 2: Quick Visualization Using Pre-computed Results

Since the repository includes pre-computed results, you can skip the GPU-intensive steps:

#### Step 1: Extract Pre-computed Data
```bash
# Extract model activations
cd raw_text_datasets/
tar -xzf model_activations_gemma-2-9b.tar.gz
tar -xzf model_activations_gemma-2-9b_OOD.tar.gz

# Extract SAE activations
tar -xzf sae_activations_gemma-2-9b_OOD.tar.gz
tar -xzf sae_activations_gemma-2-9b_OOD_1m.tar.gz

# Extract baseline results
tar -xzf baseline_results_gemma-2-9b.tar.gz
```

#### Step 2: Run Analysis Notebooks
```bash
# These notebooks work with pre-computed results
jupyter notebook plot_normal.ipynb
jupyter notebook plot_combined.ipynb
jupyter notebook plot_ood.ipynb
jupyter notebook plot_llama.ipynb
jupyter notebook sae_improvement.ipynb
```

#### Step 3: Re-run Probe Training (Optional)
```bash
# If you want to verify probe training results
python run_baselines.py --setting normal --model gemma-2-9b
python train_sae_probes.py --setting normal --model gemma-2-9b
```

## Experimental Settings

The paper tests four challenging regimes:
1. **Normal**: Standard training conditions
2. **Data Scarcity**: Limited training examples (10, 100 samples)
3. **Class Imbalance**: 90% of data from one class
4. **Label Noise**: 20% corrupted labels
5. **OOD (Covariate Shift)**: Train/test distribution mismatch

## Key Files to Understand

1. **save_sae_acts_and_train_probes.sh**: Main orchestration script
2. **combine_results.py**: Aggregates individual results into CSVs
3. **plot_*.ipynb**: Visualization notebooks for each experimental condition

## Parallelization Strategies for M3

When running probe training on M3:
```bash
# Run multiple probe training jobs in parallel
for dataset in {100..110}; do
    python train_sae_probes.py --dataset $dataset &
done
wait
```

## Expected Outputs

1. **CSV files**: Aggregated results in `results/` directory
2. **Plots**: Figures comparing SAE vs baseline performance
3. **Pickle files**: Individual probe results

## Time Estimates

- **Full reproduction with GPU**: 24-48 hours
- **Probe training only on M3**: 2-4 hours
- **Visualization from pre-computed results**: <30 minutes

## Troubleshooting

1. **Memory issues on M3**: Reduce batch sizes in probe training
2. **Missing activations**: Ensure tar files are properly extracted
3. **CUDA errors**: These scripts require GPU access; use pre-computed data instead

## Next Steps

1. Start with visualization notebooks using pre-computed results
2. If GPU access available, generate fresh activations
3. Compare your results with paper's reported findings