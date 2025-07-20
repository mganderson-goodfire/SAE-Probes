# SAE-Probes Reproduction Status

## Summary
Attempting to reproduce the paper "Are Sparse Autoencoders Useful?" on M3 MacBook Pro. Baseline probes can be trained successfully, but SAE probe training is blocked due to missing normal setting SAE activations.

## What Works
- ✅ Baseline probe training (`make train-baselines`)
- ✅ Model activations are available for all datasets
- ✅ Pre-computed probe results exist for visualization
- ✅ OOD SAE activations are available (but insufficient alone)

## What's Blocked
- ❌ SAE probe training (`train_sae_probes.py`) requires normal setting SAE activations
- ❌ Even for OOD experiments, normal activations are needed for training data

## Missing Data
Critical missing files: SAE activations for normal setting
- Expected location: `data/sae_activations_gemma-2-9b/normal_setting/`
- Format needed: Individual activation files per dataset/layer/SAE combination
  - Example: `73_control-group_20_width_16k_average_l0_408_X_train_sae.pt`

## Available Resources
1. **Model Activations** (32GB compressed)
   - `data/model_activations_gemma-2-9b.tar.gz`
   - Contains activations for layers 9, 20, 31, 41

2. **OOD SAE Activations** 
   - `data/sae_activations_gemma-2-9b_OOD.tar.gz`
   - `data/sae_activations_gemma-2-9b_OOD_1m.tar.gz`
   - Only 9 datasets, consolidated format

3. **Pre-computed Results**
   - `data/consolidated_probing_gemma-2-9b/` (461 files)
   - `raw_text_datasets/baseline_results_gemma-2-9b/`

## Recommended Actions

### For Immediate Progress (M3 Compatible)
1. **Run baseline probes**: `make train-baselines`
2. **Visualize existing results**: `make visualize`
3. **Analyze pre-computed SAE results** in `consolidated_probing_gemma-2-9b/`

### For Full Reproduction
1. **Check Dropbox** for missing normal SAE activation files
2. **Generate SAE activations** (requires GPU):
   ```bash
   python generate_sae_activations.py --model_name gemma-2-9b --setting normal --device cuda:0
   ```
3. **Use orchestration script** for complete generation:
   ```bash
   bash save_sae_acts_and_train_probes.sh
   ```

## Technical Details
- The `train_sae_probes.py` script's OOD mode still requires normal setting data for training
- OOD tar files contain consolidated results, not individual activation files needed
- Generating SAE activations requires CUDA GPU due to SAE model requirements

## Next Steps
1. Complete baseline probe training
2. Check if Dropbox contains additional SAE activation files
3. Proceed with visualization using pre-computed results
4. Document any additional findings