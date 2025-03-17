# %%
import torch
from utils_data import get_numbered_binary_tags, get_dataset_sizes, get_yvals, get_train_test_indices
from utils_sae import load_gemma_2_9b_sae
from utils_training import train_aggregated_probe_on_acts
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

model_name = "gemma-2-9b"
max_seq_len = 256
layer = 20
device = "cuda:1"
data_dir = "/mnt/sdc/jengels/data"

# Load SAE
sae_id = "layer_20/width_16k/average_l0_68"
sae = load_gemma_2_9b_sae(sae_id).to(device)

# Get datasets
datasets = get_numbered_binary_tags()
dataset_sizes = get_dataset_sizes()

# %%

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_train", type=int)
args = parser.parse_args()

# %%

# Create results dataframe
results = []
global_num_train = args.num_train

for dataset in datasets:
    # Skip if dataset is too small
    if global_num_train is None:
        num_train = min(dataset_sizes[dataset]-100, 1024)
    else:
        num_train = min(global_num_train, dataset_sizes[dataset]-100)

    # Load model activations
    try:
        X_tensor = torch.load(f"{data_dir}/model_activations_{model_name}_{max_seq_len}/{dataset}_layer_{layer}.pt", weights_only=True)
    except Exception as e:
        # print(f"Error loading {dataset}.{layer}.hook_resid_post.pt: {e}")
        continue


    # Get labels and train/test split
    y = get_yvals(dataset)
    num_test = dataset_sizes[dataset] - num_train
    train_indices, test_indices = get_train_test_indices(y, num_train, num_test, pos_ratio=0.5, seed=42)

    # Process activations through SAE in batches
    x_shape = X_tensor.shape
    flattened_x = X_tensor.flatten(end_dim=1)
    all_x_sae = []
    batch_size = 32768

    for i in tqdm(range(0, len(flattened_x), batch_size)):
        batch = flattened_x[i:i+batch_size].to(device)
        all_x_sae.append(sae.encode(batch).cpu())
    all_x_sae = torch.cat(all_x_sae)
    
    # Reshape back to sequence form
    all_x_sae = einops.rearrange(all_x_sae, "(b s) d -> b s d", b=x_shape[0], s=x_shape[1])

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

    results.append({
        "dataset": dataset,
        "metrics": metrics
    })

# %%

import pickle as pkl

with open(f"results/multi_token_data_scarcity_sae_layer20_numtrain{global_num_train}.pkl", "wb") as f:
    pkl.dump(results, f)

# %%
