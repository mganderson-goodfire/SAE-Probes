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
from sklearn.metrics import roc_auc_score
os.environ["OMP_NUM_THREADS"] = "8"
import pickle as pkl

warnings.simplefilter("ignore", category=ConvergenceWarning)

model_name = "gemma-2-9b"
max_seq_len = 256
layer = 20
device = "cuda:1"
data_dir = "/mnt/sdc/jengels/data"

# Get datasets
datasets = get_numbered_binary_tags()
dataset_sizes = get_dataset_sizes()

# %%

def train_attn_probing(X_train, X_test, y_train, y_test, l2_lambda = 0):
    """Train a simple attention-based probe on the data with learning"""
    # Get sequence length and feature dimension
    seq_len = X_train.shape[1]
    feat_dim = X_train.shape[2]
    
    # Convert to torch tensors if needed
    X_train = torch.tensor(X_train) if not isinstance(X_train, torch.Tensor) else X_train
    X_test = torch.tensor(X_test) if not isinstance(X_test, torch.Tensor) else X_test
    y_train = torch.tensor(y_train) if not isinstance(y_train, torch.Tensor) else y_train
    y_test = torch.tensor(y_test) if not isinstance(y_test, torch.Tensor) else y_test

    # Split train into train/val
    n_train = int(0.8 * len(X_train))
    indices = torch.randperm(len(X_train))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    X_train_split = X_train[train_idx].to(device)
    y_train_split = y_train[train_idx].to(device)

    # Initialize learnable parameters
    output_vector = torch.nn.Parameter(torch.randn(feat_dim).to(device) * 0.001)
    attn_vector = torch.nn.Parameter(torch.randn(feat_dim).to(device) * 0.001)
    bias = torch.nn.Parameter(torch.zeros(1).to(device))
    optimizer = torch.optim.Adam([output_vector, attn_vector, bias], lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Track best validation performance
    best_val_auc = 0
    best_output_vector = None
    best_attn_vector = None 
    best_bias = None

    val_vectors = X_train[val_idx].to(device)
    pbar = tqdm(range(1000))
    for epoch in pbar:
        # Training step
        optimizer.zero_grad()

        attn_scores = torch.matmul(X_train_split, attn_vector)
        # print(attn_scores)
        # Replace exactly zeros with -10000
        with torch.no_grad():
            attn_scores[attn_scores == 0] = -10000
        attn_weights = torch.softmax(attn_scores, dim=1)
        weighted_sum = einops.einsum(attn_weights, X_train_split, "batch seq_len, batch seq_len d -> batch d") 
        logits = torch.matmul(weighted_sum, output_vector) + bias
        
        # Forward pass on training data
        # logits = torch.matmul(last_vectors, output_vector) + bias  # Shape: [batch_size]

        # Calculate loss with L2 regularization
        l2_reg = l2_lambda * (torch.norm(output_vector) ** 2 + torch.norm(attn_vector) ** 2)
        loss = criterion(logits, y_train_split.float()) + l2_reg
        
        # Backward pass
        loss.backward()

        # Calculate train AUC
        train_auc = roc_auc_score(y_train_split.detach().cpu().numpy(), logits.detach().cpu().numpy())

        # Calculate validation AUC
        with torch.no_grad():
            val_attn_scores = torch.matmul(val_vectors, attn_vector)
            val_attn_scores[val_attn_scores == 0] = -10000
            val_attn_weights = torch.softmax(val_attn_scores, dim=1)
            val_weighted_sum = einops.einsum(val_attn_weights, val_vectors, "batch seq_len, batch seq_len d -> batch d")
            val_logits = torch.matmul(val_weighted_sum, output_vector) + bias
            val_auc = roc_auc_score(y_train[val_idx].cpu().numpy(), val_logits.cpu().numpy())
            
            # Save best parameters
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_output_vector = output_vector.clone().detach()
                best_attn_vector = attn_vector.clone().detach()
                best_bias = bias.clone().detach()
        
        optimizer.step()

        # Update progress bar with train and validation AUC
        pbar.set_description(
            f"Train AUC: {train_auc:.3f}, "
            f"Val AUC: {val_auc:.3f}, "
            f"L: {loss.item():.3f}, "
            f"B: {bias.item():.3f}, "
            f"A_N: {torch.norm(attn_vector).item():.3f}, "
            f"O_N: {torch.norm(output_vector).item():.3f}, "
            f"A_NZ: {(attn_weights > 0).float().mean().item():.3f}"
        )

    # Calculate test AUC using best parameters
    with torch.no_grad():
        test_attn_scores = torch.matmul(X_test.cpu(), best_attn_vector.cpu())
        test_attn_scores[test_attn_scores == 0] = -10000
        test_attn_weights = torch.softmax(test_attn_scores, dim=1)
        test_weighted_sum = einops.einsum(test_attn_weights, X_test.cpu(), "batch seq_len, batch seq_len d -> batch d")
        test_logits = torch.matmul(test_weighted_sum, best_output_vector.cpu()) + best_bias.cpu()
        test_auc = roc_auc_score(y_test.cpu().numpy(), test_logits.numpy())

    metrics = {
        "train_auc": train_auc,
        "val_auc": best_val_auc,
        "test_auc": test_auc,
        "best_output_vector": best_output_vector,
        "best_attn_vector": best_attn_vector,
        "best_bias": best_bias
    }
    return metrics

def train_attn_probing_on_model_acts(dataset, layer, num_train=None):

    X_tensor = torch.load(f"{data_dir}/model_activations_{model_name}_{max_seq_len}/{dataset}_layer_{layer}.pt", weights_only=True).float()

    size = dataset_sizes[dataset]
    if num_train is None:
        num_train = min(size-100, 1024)
    num_test = size - num_train
    y = get_yvals(dataset)
    train_indices, test_indices = get_train_test_indices(y, num_train, num_test, pos_ratio=0.5, seed=42)

    y_train = y[train_indices]
    y_test = y[test_indices]

    X_train_model = X_tensor[train_indices]
    X_test_model = X_tensor[test_indices]

    metrics = train_attn_probing(X_train_model, X_test_model, y_train, y_test)
    metrics["dataset"] = dataset
    metrics["layer"] = layer

    return metrics


all_metrics = []
global_num_train = None
for dataset in datasets:
    metrics = train_attn_probing_on_model_acts(dataset, layer, global_num_train)
    all_metrics.append(metrics)

if global_num_train is None:
    with open("results/multi_token_baseline_layer20_numtrainall.pkl", "wb") as f:
        pkl.dump(all_metrics, f)
else:
    with open(f"results/multi_token_baseline_layer20_numtrain{global_num_train}.pkl", "wb") as f:
        pkl.dump(all_metrics, f)




# %%
