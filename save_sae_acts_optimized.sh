#!/bin/bash

# Optimized script that processes all datasets per SAE load
# This is MUCH faster than the original approach

echo "Starting optimized SAE activation generation..."

# ---- Generate SAE activations using batch mode ----

# Normal setting - single run processes ALL datasets for each SAE
echo "Processing normal setting..."
python3 generate_sae_activations_optimized.py \
    --model_name gemma-2-9b \
    --setting normal \
    --device mps \
    --batch_mode

# Scarcity setting
echo "Processing scarcity setting..."
python3 generate_sae_activations_optimized.py \
    --model_name gemma-2-9b \
    --setting scarcity \
    --device mps \
    --batch_mode

# Imbalance setting
echo "Processing imbalance setting..."
python3 generate_sae_activations_optimized.py \
    --model_name gemma-2-9b \
    --setting imbalance \
    --device mps \
    --batch_mode

# OOD setting
echo "Processing OOD setting..."
python3 generate_sae_activations_optimized.py \
    --model_name gemma-2-9b \
    --setting OOD \
    --device mps \
    --batch_mode

echo "Optimized SAE activation generation complete!"

# ---- Train SAE probes (unchanged) ----

# Uncomment to also train probes
# for i in {1..20}
# do
#     OMP_NUM_THREADS=1 python3 train_sae_probes.py --model_name gemma-2-9b --setting normal --reg_type l1 --randomize_order &
# done
# wait

# ... rest of probe training commands ...