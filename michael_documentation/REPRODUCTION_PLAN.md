# Reproduction Plan: "Are Sparse Autoencoders Useful?"

This plan outlines how to reproduce the findings from the paper, with specific consideration for running on an M3 MacBook Pro (Apple Silicon) versus GPU clusters.

## Overview

The reproduction workflow has three main stages:
1. **Activation Generation** (GPU-intensive)
2. **Probe Training** (CPU-friendly)
3. **Visualization & Analysis** (CPU-friendly)

## Hardware Requirements

### M3 MacBook Pro Capabilities
- ✅ Can run: Probe training, results analysis, visualization
- ❌ Cannot run: Model/SAE activation generation (requires CUDA GPUs)
- Memory: Sufficient for probe training and analysis tasks

### GPU Cluster Requirements
- CUDA-capable GPUs with 18GB+ VRAM for 9B parameter models
- Multiple GPUs recommended for parallel activation generation

## Reproduction Steps

### Option 1: Full Reproduction (Requires GPU Access)

#### Step 1: Environment Setup
```bash
# Create virtual environment
python -m venv probing
source probing/bin/activate

# Install dependencies
pip install transformer_lens sae_lens transformers datasets torch xgboost sae_bench scikit-learn natsort
```

#### Step 2: Generate Model Activations (GPU Required)
```bash
# On GPU cluster
python generate_model_activations.py

# This generates activations for:
# - gemma-2-9b (layers 20, 31, 41)
# - llama-3.1-8b (layers 15, 23, 31) 
# - gemma-2-2b (layers 12, 18, 25)
```

#### Step 3: Generate SAE Activations (GPU Required)
```bash
# Run the orchestration script (uses cuda:1)
bash save_sae_acts_and_train_probes.sh

# Or run individually:
python generate_sae_activations.py --setting normal --model gemma-2-9b
```

#### Step 4: Train Probes (Can Run on M3)
```bash
# Baseline probes
python run_baselines.py --setting normal --model gemma-2-9b

# SAE probes (already handled by orchestration script)
python train_sae_probes.py --setting normal --model gemma-2-9b
```

#### Step 5: Combine Results (Can Run on M3)
```bash
python combine_results.py
```

#### Step 6: Generate Visualizations (Can Run on M3)
```bash
# Run Jupyter notebooks for plots
jupyter notebook plot_normal.ipynb
jupyter notebook plot_combined.ipynb
jupyter notebook plot_ood.ipynb
```

### Option 2: Partial Reproduction Using Pre-computed Activations (M3-Friendly)

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