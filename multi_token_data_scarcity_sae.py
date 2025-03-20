# %%
import torch
from utils_data import get_numbered_binary_tags, get_dataset_sizes, get_yvals, get_train_test_indices, get_training_sizes
from utils_sae import load_gemma_2_9b_sae
from utils_training import train_aggregated_probe_on_acts
from handle_sae_bench_saes import *
import os
from tqdm import tqdm
import pandas as pd
from utils_training import find_best_reg
import pickle as pkl
import warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import einops
os.environ["OMP_NUM_THREADS"] = "8"

warnings.simplefilter("ignore", category=ConvergenceWarning)
torch.set_grad_enabled(False)

max_seq_len = 256
device = "cuda:1"
data_dir = "/mnt/sdc/jengels/data"

# Load SAE
model_name = "gemma-2-2b"
layer = 12
sae_ids = get_sae_ids_closest_to_target_l0(layer, 100)
saes = [load_gemma_2_2b_sae(sae_id).to(device) for sae_id in sae_ids]

# Load SAE
# model_name = "gemma-2-9b"
# layer = 20
# sae_id = "layer_20/width_16k/average_l0_68"
# sae = load_gemma_2_9b_sae(sae_id).to(device)

# Get datasets
datasets = get_numbered_binary_tags()
dataset_sizes = get_dataset_sizes()
# training_sizes = get_training_sizes()


# training_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
training_sizes = [100, 1024]

# %%

# Create results dataframe

for dataset in tqdm(datasets):
    results_file_path = f"data/multi_token_scarcity/{dataset}_all_saes_{model_name}_{layer}.pkl"
    if os.path.exists(results_file_path):
        continue
    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

    # Load model activations
    try:
        file_path = f"{data_dir}/model_activations_{model_name}_{max_seq_len}/{dataset}_layer_{layer}.pt"
        print(f"Loading {file_path}")
        X_tensor = torch.load(file_path, weights_only=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        continue


    # Get labels and train/test split
    y = get_yvals(dataset)

    # Process activations through SAE in batches
    x_shape = X_tensor.shape
    flattened_x = X_tensor.flatten(end_dim=1)

    results = []

    for sae_id, sae in zip(sae_ids, saes):
        all_x_sae = []
        batch_size = 32768

        for i in range(0, len(flattened_x), batch_size):
            batch = flattened_x[i:i+batch_size].to(device)
            all_x_sae.append(sae.encode(batch.float()).cpu())
        all_x_sae = torch.cat(all_x_sae)
        
        # Reshape back to sequence form
        all_x_sae = einops.rearrange(all_x_sae, "(b s) d -> b s d", b=x_shape[0], s=x_shape[1])

        for target_num_train in training_sizes:

            num_train = min(target_num_train, dataset_sizes[dataset]-102)
            num_test = dataset_sizes[dataset] - num_train - 2
            train_indices, test_indices = get_train_test_indices(y, num_train, num_test, pos_ratio=0.5, seed=42)

            # Split into train and test
            X_train_sae = all_x_sae[train_indices]
            X_test_sae = all_x_sae[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]

            # Try different aggregation methods
            metrics = train_aggregated_probe_on_acts(
                X_train=X_train_sae,
                X_test=X_test_sae, 
                y_train=y_train,
                y_test=y_test,
                aggregation_method="max",
                k=128,
                binarize=False
            )

            metrics["dataset"] = dataset
            metrics["sae_id"] = sae_id
            metrics["num_train"] = target_num_train
            metrics["actual_num_train"] = num_train
            results.append(metrics)

    with open(results_file_path, "wb") as f:
        pkl.dump(results, f)

