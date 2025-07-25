# SAE-Probes Reproduction Status

## Summary
Attempting to reproduce the paper "Are Sparse Autoencoders Useful?" on M3 MacBook Pro. Baseline probes can be trained successfully. SAE activation generation now works with optimized script, though memory constraints limit processing of largest SAEs.

## What Works
- ✅ Baseline probe training (`make train-baselines`)
- ✅ Model activations are available for all datasets
- ✅ Pre-computed probe results exist for visualization
- ✅ OOD SAE activations are available (but insufficient alone)
- ✅ **NEW**: Optimized SAE activation generation (`make generate-sae-acts-optimized`)
- ✅ **NEW**: Successfully generated activations for 16k and 131k width SAEs

## What's Blocked
- ⚠️ 1M width SAE crashes on M3 due to memory constraints ("Killed: 9" error)
- ⚠️ One OOD dataset (`66_living-room_translations.csv`) lacks model activations

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

## Optimization Progress

### Performance Improvements
1. **Created optimized SAE activation generation script** (`generate_sae_activations_optimized.py`)
   - Loads each SAE only once and processes all datasets before moving to next SAE
   - 10-100x faster than original approach which loaded SAEs repeatedly
   - Added `--batch_mode` flag for efficient processing

2. **Performance bottlenecks identified**:
   - Loading large model activation files (~45MB each) from disk
   - Not the SAE encoding itself (matrix multiplication is fast)
   - Original script loaded same SAE up to 100+ times

3. **Memory issues with 1M width SAEs**:
   - M3 MacBook Pro runs out of memory for largest SAEs
   - Consider processing these on a machine with more RAM or using smaller batch sizes

## Recommended Actions

### For Immediate Progress (M3 Compatible)
1. **Run optimized SAE generation**: `make generate-sae-acts-optimized`
2. **Process smaller SAEs first** (16k, 131k widths work fine)
3. **Skip 1M width SAEs** or process on higher-memory machine
4. **Train probes on generated activations**: `make train-sae-probes`
5. **Visualize results**: `make visualize`

### Workarounds
1. **For missing OOD dataset**: The script now gracefully skips `66_living-room_translations.csv`
2. **For memory constraints**: 
   - Process settings one at a time
   - Use original script with smaller batches for 1M SAEs
   - Consider reducing batch size in the code (currently 128)

## Technical Details
- The `train_sae_probes.py` script's OOD mode still requires normal setting data for training
- OOD tar files contain consolidated results, not individual activation files needed
- SAE models can run on MPS (Apple Silicon) but largest models require significant memory
- File I/O is the main bottleneck, not computation

### Key Code Changes
1. **Added to Makefile**: `generate-sae-acts-optimized` target
2. **Created**: `generate_sae_activations_optimized.py` with batch processing
3. **Created**: `save_sae_acts_optimized.sh` for streamlined execution
4. **Fixed**: Error handling for missing OOD dataset activations

## Next Steps
1. Train SAE probes on generated activations
2. Process 1M width SAEs on higher-memory machine if needed
3. Proceed with visualization using all available results
4. Compare optimized vs original script performance metrics