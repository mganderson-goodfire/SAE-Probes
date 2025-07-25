# %%
import torch
import os
import argparse
import warnings
from tqdm import tqdm
import random
from sklearn.exceptions import ConvergenceWarning
from utils_data import (
    get_OOD_datasets,
    get_dataset_sizes, 
    get_numbered_binary_tags, 
    get_xy_traintest, 
    get_xy_traintest_specify,
    get_training_sizes,
    get_class_imbalance,
    get_classimabalance_num_train,
)
from utils_sae import (
    layer_to_sae_ids, 
    sae_id_to_sae, 
    get_sae_layers,
    get_sae_layers_extra
)

warnings.simplefilter("ignore", category=ConvergenceWarning)
torch.set_grad_enabled(False)

# %%

# Common variables
dataset_sizes = get_dataset_sizes()
datasets = get_numbered_binary_tags()

# %%
# Helper functions for all settings
def save_activations(path, activation):
    """Save activations in sparse format to save space"""
    sparse_tensor = activation.to_sparse()
    torch.save(sparse_tensor, path)

def load_activations(path):
    """Load activations from sparse format"""
    return torch.load(path, weights_only=True).to_dense().float()

# %%
# Normal setting functions
def get_sae_paths_normal(dataset, layer, sae_id, model_name="gemma-2-9b"):
    """Get paths for normal setting"""
    os.makedirs(f"data/sae_probes_{model_name}/normal_setting", exist_ok=True)
    os.makedirs(f"data/sae_activations_{model_name}/normal_setting", exist_ok=True)

    if model_name == "gemma-2-9b":
        width = sae_id.split("/")[1]
        l0 = sae_id.split("/")[2]
        description_string = f"{dataset}_{layer}_{width}_{l0}"
    elif model_name == "llama-3.1-8b":
        description_string = f"{dataset}_{sae_id}"
    elif model_name == "gemma-2-2b":
        name = '_'.join(sae_id[2].split('/')[0].split('_')[1:])
        l0 = sae_id[3]
        rounded_l0 = round(float(l0))
        description_string = f"{dataset}_{name}_{rounded_l0}"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    train_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_X_train_sae.pt"
    test_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_X_test_sae.pt"
    y_train_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_y_train.pt"
    y_test_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_y_test.pt"
    return {
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path
    }

def process_dataset_normal(dataset, layer, sae, sae_id, model_name, device):
    """Process a single dataset for normal setting"""
    paths = get_sae_paths_normal(dataset, layer, sae_id, model_name)
    train_path, test_path, y_train_path, y_test_path = paths["train_path"], paths["test_path"], paths["y_train_path"], paths["y_test_path"]
    
    if all([os.path.exists(train_path), os.path.exists(test_path), os.path.exists(y_train_path), os.path.exists(y_test_path)]):
        return False
    
    size = dataset_sizes[dataset]
    num_train = min(size-100, 1024)
    X_train, y_train, X_test, y_test = get_xy_traintest(num_train, dataset, layer, model_name=model_name)

    batch_size = 128
    X_train_sae = []
    for i in range(0, len(X_train), batch_size):
        batch = X_train[i:i+batch_size].to(device)
        X_train_sae.append(sae.encode(batch).cpu())
    X_train_sae = torch.cat(X_train_sae)

    X_test_sae = []
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size].to(device)
        X_test_sae.append(sae.encode(batch).cpu())
    X_test_sae = torch.cat(X_test_sae)

    save_activations(train_path, X_train_sae)
    save_activations(test_path, X_test_sae)
    save_activations(y_train_path, torch.tensor(y_train))
    save_activations(y_test_path, torch.tensor(y_test))
    
    return True

# %%
# Data scarcity setting functions
def get_sae_paths_scarcity(dataset, layer, sae_id, num_train, model_name="gemma-2-9b"):
    """Get paths for data scarcity setting"""
    os.makedirs(f"data/sae_probes_{model_name}/scarcity_setting", exist_ok=True)
    os.makedirs(f"data/sae_activations_{model_name}/scarcity_setting", exist_ok=True)
    
    if model_name == "gemma-2-9b":
        width = sae_id.split("/")[1]
        l0 = sae_id.split("/")[2]
        description_string = f"{dataset}_{layer}_{width}_{l0}"
    elif model_name == "llama-3.1-8b":
        description_string = f"{dataset}_{sae_id}"
    elif model_name == "gemma-2-2b":
        name = '_'.join(sae_id[2].split('/')[0].split('_')[1:])
        l0 = sae_id[3]
        rounded_l0 = round(float(l0))
        description_string = f"{dataset}_{name}_{rounded_l0}"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    train_path = f"data/sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_X_train_sae.pt"
    test_path = f"data/sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_X_test_sae.pt"
    y_train_path = f"data/sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_y_train.pt"
    y_test_path = f"data/sae_activations_{model_name}/scarcity_setting/{description_string}_{num_train}_y_test.pt"
    
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    
    return {
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path
    }

def process_dataset_scarcity(dataset, layer, sae, sae_id, model_name, device):
    """Process a single dataset for data scarcity setting"""
    train_sizes = get_training_sizes()
    processed_any = False
    
    for num_train in train_sizes:
        if num_train > dataset_sizes[dataset] - 100:
            continue
            
        paths = get_sae_paths_scarcity(dataset, layer, sae_id, num_train, model_name)
        train_path, test_path, y_train_path, y_test_path = paths["train_path"], paths["test_path"], paths["y_train_path"], paths["y_test_path"]
        
        if all(os.path.exists(p) for p in [train_path, test_path, y_train_path, y_test_path]):
            continue

        X_train, y_train, X_test, y_test = get_xy_traintest(num_train, dataset, layer, model_name=model_name)

        batch_size = 128
        X_train_sae = []
        for i in range(0, len(X_train), batch_size):
            batch = X_train[i:i+batch_size].to(device)
            X_train_sae.append(sae.encode(batch).cpu())
        X_train_sae = torch.cat(X_train_sae)

        X_test_sae = []
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size].to(device)
            X_test_sae.append(sae.encode(batch).cpu())
        X_test_sae = torch.cat(X_test_sae)

        save_activations(train_path, X_train_sae)
        save_activations(test_path, X_test_sae)
        save_activations(y_train_path, torch.tensor(y_train))
        save_activations(y_test_path, torch.tensor(y_test))
        processed_any = True
    
    return processed_any

# %%
# Class imbalance setting functions
def get_sae_paths_imbalance(dataset, layer, sae_id, frac, model_name="gemma-2-9b"):
    """Get paths for class imbalance setting"""
    os.makedirs(f"data/sae_probes_{model_name}/class_imbalance_setting", exist_ok=True)
    os.makedirs(f"data/sae_activations_{model_name}/class_imbalance_setting", exist_ok=True)
    
    if model_name == "gemma-2-9b":
        width = sae_id.split("/")[1]
        l0 = sae_id.split("/")[2]
        description_string = f"{dataset}_{layer}_{width}_{l0}"
    elif model_name == "llama-3.1-8b":
        description_string = f"{dataset}_{sae_id}"
    elif model_name == "gemma-2-2b":
        name = '_'.join(sae_id[2].split('/')[0].split('_')[1:])
        l0 = sae_id[3]
        rounded_l0 = round(float(l0))
        description_string = f"{dataset}_{name}_{rounded_l0}"
    else:
        raise ValueError(f"Invalid model name: {model_name}")
        
    train_path = f"data/sae_activations_{model_name}/class_imbalance_setting/{description_string}_frac{frac}_X_train_sae.pt"
    test_path = f"data/sae_activations_{model_name}/class_imbalance_setting/{description_string}_frac{frac}_X_test_sae.pt"
    y_train_path = f"data/sae_activations_{model_name}/class_imbalance_setting/{description_string}_frac{frac}_y_train.pt"
    y_test_path = f"data/sae_activations_{model_name}/class_imbalance_setting/{description_string}_frac{frac}_y_test.pt"
    return {
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path
    }

def process_dataset_imbalance(dataset, layer, sae, sae_id, model_name, device):
    """Process a single dataset for class imbalance setting"""
    fracs = get_class_imbalance()
    processed_any = False
    
    for frac in fracs:
        paths = get_sae_paths_imbalance(dataset, layer, sae_id, frac, model_name)
        train_path, test_path, y_train_path, y_test_path = paths["train_path"], paths["test_path"], paths["y_train_path"], paths["y_test_path"]
        
        if os.path.exists(train_path):
            continue
        
        num_train, num_test = get_classimabalance_num_train(dataset)
        X_train, y_train, X_test, y_test = get_xy_traintest_specify(
            num_train, dataset, layer, pos_ratio=frac, model_name=model_name, num_test=num_test
        )

        batch_size = 128
        X_train_sae = []
        for i in range(0, len(X_train), batch_size):
            batch = X_train[i:i+batch_size].to(device)
            X_train_sae.append(sae.encode(batch).cpu())
        X_train_sae = torch.cat(X_train_sae)

        X_test_sae = []
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size].to(device)
            X_test_sae.append(sae.encode(batch).cpu())
        X_test_sae = torch.cat(X_test_sae)

        save_activations(train_path, X_train_sae)
        save_activations(test_path, X_test_sae)
        save_activations(y_train_path, torch.tensor(y_train))
        save_activations(y_test_path, torch.tensor(y_test))
        processed_any = True
    
    return processed_any

# %%
# OOD setting functions
def get_sae_paths_ood(dataset, layer, sae_id, model_name="gemma-2-9b"):
    os.makedirs(f"data/sae_probes_{model_name}/ood_setting", exist_ok=True)
    os.makedirs(f"data/sae_activations_{model_name}/ood_setting", exist_ok=True)

    if model_name == "gemma-2-9b":
        width = sae_id.split("/")[1]
        l0 = sae_id.split("/")[2]
        description_string = f"{dataset}_{layer}_{width}_{l0}"
    elif model_name == "llama-3.1-8b":
        description_string = f"{dataset}_{sae_id}"
    elif model_name == "gemma-2-2b":
        name = '_'.join(sae_id[2].split('/')[0].split('_')[1:])
        l0 = sae_id[3]
        rounded_l0 = round(float(l0))
        description_string = f"{dataset}_{name}_{rounded_l0}"
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    train_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_X_train_sae.pt"
    test_path = f"data/sae_activations_{model_name}/OOD_setting/{description_string}_X_test_sae.pt"
    y_train_path = f"data/sae_activations_{model_name}/normal_setting/{description_string}_y_train.pt"
    y_test_path = f"data/sae_activations_{model_name}/OOD_setting/{description_string}_y_test.pt"
    return {
        "train_path": train_path,
        "test_path": test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path
    }

def process_dataset_ood(dataset, layer, sae, sae_id, model_name, device):
    """Process a single dataset for OOD setting"""
    paths = get_sae_paths_ood(dataset, layer, sae_id, model_name)
    train_path, test_path, y_train_path, y_test_path = paths["train_path"], paths["test_path"], paths["y_train_path"], paths["y_test_path"]
    
    if all(os.path.exists(p) for p in [train_path, test_path, y_train_path, y_test_path]):
        return False
    
    try:
        X_train, y_train, X_test, y_test = get_xy_traintest(1024, dataset, layer, model_name=model_name)
    except FileNotFoundError as e:
        print(f"  Warning: Skipping dataset {dataset} - model activations not found")
        return False
    
    batch_size = 128
    X_train_sae = []
    for i in range(0, len(X_train), batch_size):
        batch = X_train[i:i+batch_size].to(device)
        X_train_sae.append(sae.encode(batch).cpu())
    X_train_sae = torch.cat(X_train_sae)

    X_test_sae = []
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size].to(device)
        X_test_sae.append(sae.encode(batch).cpu())
    X_test_sae = torch.cat(X_test_sae)

    save_activations(train_path, X_train_sae)
    save_activations(test_path, X_test_sae)
    save_activations(y_train_path, torch.tensor(y_train))
    save_activations(y_test_path, torch.tensor(y_test))
    
    return True

# %%
# Process SAEs for a specific model and setting
def process_model_setting(model_name, setting, device, randomize_order, batch_mode=False):
    print(f"Running SAE activation generation for {model_name} in {setting} setting")
    if batch_mode:
        print("Batch mode enabled: will process all datasets per SAE load")
    
    layers = get_sae_layers(model_name)
    if model_name == "gemma-2-9b" and setting == "normal":
        layers = get_sae_layers_extra(model_name)
    
    total_processed = 0
    
    for layer in layers:
        sae_ids = layer_to_sae_ids(layer, model_name)
        if model_name == "gemma-2-9b" and setting != "normal":
            sae_ids = ["layer_20/width_16k/average_l0_408", "layer_20/width_131k/average_l0_276", "layer_20/width_1m/average_l0_193"]
            
        if randomize_order:
            random.shuffle(sae_ids)
        
        for sae_id in sae_ids:
            print(f"\nProcessing SAE: {sae_id}")
            
            # First check if this SAE needs any processing
            needs_processing = False
            
            if setting == "normal":
                datasets_to_process = datasets if not setting == "OOD" else get_OOD_datasets()
                for dataset in datasets_to_process:
                    paths = get_sae_paths_normal(dataset, layer, sae_id, model_name)
                    if not all(os.path.exists(p) for p in [paths["train_path"], paths["test_path"], 
                                                          paths["y_train_path"], paths["y_test_path"]]):
                        needs_processing = True
                        break
                        
            elif setting == "scarcity":
                train_sizes = get_training_sizes()
                for dataset in datasets:
                    for num_train in train_sizes:
                        if num_train > dataset_sizes[dataset] - 100:
                            continue
                        paths = get_sae_paths_scarcity(dataset, layer, sae_id, num_train, model_name)
                        if not all(os.path.exists(p) for p in [paths["train_path"], paths["test_path"], 
                                                              paths["y_train_path"], paths["y_test_path"]]):
                            needs_processing = True
                            break
                    if needs_processing:
                        break
                        
            elif setting == "imbalance":
                fracs = get_class_imbalance()
                for dataset in datasets:
                    for frac in fracs:
                        paths = get_sae_paths_imbalance(dataset, layer, sae_id, frac, model_name)
                        if not all(os.path.exists(p) for p in [paths["train_path"], paths["test_path"], 
                                                              paths["y_train_path"], paths["y_test_path"]]):
                            needs_processing = True
                            break
                    if needs_processing:
                        break
                        
            elif setting == "OOD":
                for dataset in get_OOD_datasets():
                    paths = get_sae_paths_ood(dataset, layer, sae_id, model_name)
                    if not all(os.path.exists(p) for p in [paths["train_path"], paths["test_path"], 
                                                          paths["y_train_path"], paths["y_test_path"]]):
                        needs_processing = True
                        break
            
            if not needs_processing:
                print(f"  All datasets already processed for this SAE")
                continue
            
            # Load the SAE
            try:
                sae = sae_id_to_sae(sae_id, model_name, device)
                print(f"  Loaded SAE successfully")
            except Exception as e:
                print(f"  Error loading SAE {sae_id}: {e}")
                continue
            
            # Process all datasets that need this SAE
            sae_processed_count = 0
            
            if setting == "normal":
                datasets_to_process = datasets
                for dataset in tqdm(datasets_to_process, desc="Processing datasets"):
                    if process_dataset_normal(dataset, layer, sae, sae_id, model_name, device):
                        sae_processed_count += 1
                        if not batch_mode:
                            break
                            
            elif setting == "scarcity":
                for dataset in tqdm(datasets, desc="Processing datasets"):
                    if process_dataset_scarcity(dataset, layer, sae, sae_id, model_name, device):
                        sae_processed_count += 1
                        if not batch_mode:
                            break
                            
            elif setting == "imbalance":
                for dataset in tqdm(datasets, desc="Processing datasets"):
                    if process_dataset_imbalance(dataset, layer, sae, sae_id, model_name, device):
                        sae_processed_count += 1
                        if not batch_mode:
                            break
                            
            elif setting == "OOD":
                for dataset in tqdm(get_OOD_datasets(), desc="Processing OOD datasets"):
                    if process_dataset_ood(dataset, layer, sae, sae_id, model_name, device):
                        sae_processed_count += 1
                        if not batch_mode:
                            break
            
            print(f"  Processed {sae_processed_count} datasets for this SAE")
            total_processed += sae_processed_count
            
            # Clear the SAE from memory
            del sae
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # In non-batch mode, return after processing one dataset
            if not batch_mode and sae_processed_count > 0:
                return True
    
    if total_processed == 0:
        print(f"All SAE activations for {model_name} in {setting} setting have been generated!")
    else:
        print(f"Total datasets processed: {total_processed}")
    
    return total_processed > 0

# %%
# Main function to process all models and settings
if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_name", type=str, default=None, choices=["gemma-2-9b", "llama-3.1-8b", "gemma-2-2b"])
    parser.add_argument("--setting", type=str, default=None, 
                        choices=["normal", "scarcity", "imbalance", "OOD"])
    parser.add_argument("--randomize_order", action="store_true", help="Randomize the order of datasets and settings, useful for parallelizing")
    parser.add_argument("--batch_mode", action="store_true", help="Process all datasets for each SAE before moving to the next (more efficient)")

    args = parser.parse_args()
    device = args.device
    model_name = args.model_name
    setting = args.setting
    randomize_order = args.randomize_order
    batch_mode = args.batch_mode

    model_names = ["gemma-2-9b", "llama-3.1-8b", "gemma-2-2b"]
    settings = ["normal", "scarcity", "imbalance", "OOD"]
    
    # If specific model and setting are provided via command line, use those
    if model_name is not None and setting is not None:
        if batch_mode:
            # In batch mode, process everything in one go
            process_model_setting(model_name, setting, device, randomize_order, batch_mode=True)
        else:
            # In non-batch mode, keep the original behavior for compatibility
            do_loop = True
            while do_loop:
                do_loop = process_model_setting(model_name, setting, device, randomize_order, batch_mode=False)
        exit(0)

    # Otherwise, loop through all models and settings
    for curr_model_name in model_names:
        if randomize_order:
            random.shuffle(settings)
        for curr_setting in settings:
            print(f"\n{'='*50}")
            print(f"Processing {curr_model_name} in {curr_setting} setting")
            print(f"{'='*50}\n")
            if batch_mode:
                process_model_setting(curr_model_name, curr_setting, device, randomize_order, batch_mode=True)
            else:
                do_loop = True
                while do_loop:
                    do_loop = process_model_setting(curr_model_name, curr_setting, device, randomize_order, batch_mode=False)