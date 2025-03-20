# %%
import glob
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# %%
def process_metrics(file, model_name):
    with open(file, "rb") as f:
        try:
            metrics = pickle.load(f)
            if model_name == "gemma-2-2b":
                for metric in metrics:
                    sae_id = metric["sae_id"]
                    name = '_'.join(sae_id[2].split('/')[0].split('_')[1:])
                    l0 = sae_id[3]
                    rounded_l0 = round(float(l0))
                    metric["sae_id"] = sae_id[2]
                    metric["sae_l0"] = rounded_l0
            return metrics
        except Exception as e:
            return None

def process_files(files, model_name):
    all_metrics = []
    bad_files = []
    
    file_iterator = tqdm(files)
    
    for file in file_iterator:
        metrics = process_metrics(file, model_name)
        if metrics:
            all_metrics.append(metrics)
        else:
            bad_files.append(file)
    
    return all_metrics, bad_files

def extract_sae_features(df, model_name):
    if model_name == "gemma-2-9b":
        df.loc[:, "sae_width"] = df["sae_id"].apply(lambda x: x.split("/")[1].split("_")[1])
        df.loc[:, "sae_l0"] = df["sae_id"].apply(lambda x: int(x.split("/")[2].split("_")[2]))
    return df

def process_setting(setting, model_name):
    print(f"Processing {setting} setting for {model_name}...")
    
    # Create output directory
    output_dir = f"results/sae_probes_{model_name}/{setting}_setting"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get file pattern based on setting
    file_pattern = f"data/sae_probes_{model_name}/{setting}_setting/*.pkl"
    
    # Process files
    files = glob.glob(file_pattern)
    print(file_pattern)
    print(len(files))
    all_metrics, bad_files = process_files(files, model_name)
    assert len(bad_files) == 0, f"Found {len(bad_files)} bad files in {setting} setting"
    
    # Create dataframe
    df = pd.DataFrame([item for sublist in all_metrics for item in sublist])
    
    # Save to CSV
    df.to_csv(f"{output_dir}/all_metrics.csv", index=False)
        
    # Print dataset length
    print(f"Total records in {setting} setting: {len(df)}")
        
    
    return df

# %%

for setting in ["normal", "scarcity", "class_imbalance", "label_noise"]:
    for model_name in ["gemma-2-9b", "llama-3.1-8b", "gemma-2-2b"]:
        process_setting(setting, model_name)
# %%

# Process multi token data scarcity sae results
all_data = []
files = glob.glob("data/multi_token_scarcity/*.pkl")
for file in files:
    with open(file, "rb") as f:
        data = pickle.load(f)
        for entry in data:
            all_data.append({
                "dataset": entry["dataset"],
                "test_f1": entry["test_f1"],
                "test_auc": entry["test_auc"],
                "val_auc": entry["val_auc"],
                "sae_id": entry["sae_id"][2],
                "num_train": entry["num_train"],
                "model_name": entry["sae_id"][1]
                })
            
os.makedirs("results/sae_probes_gemma-2-2b/multi_token_scarcity_setting", exist_ok=True)
df = pd.DataFrame(all_data)
df.to_csv("results/sae_probes_gemma-2-2b/multi_token_scarcity_setting/all_metrics.csv", index=False)


# %%

# Process multi token data scarcity baseline results
all_data = pickle.load(open("results/multi_token_baseline_layer20_numtrainall.pkl", "rb"))
pandas_data = []
for entry in all_data:
    res = {
        "dataset": entry["dataset"],
        "test_auc": entry["test_auc"],
        "val_auc": entry["val_auc"],
        "num_train": entry["num_train"],
        "model_name": "gemma-2-2b"
    }

    if res["num_train"] > 100:
        res["num_train"] = 1024
    pandas_data.append(res)
os.makedirs("results/baseline_probes_gemma-2-2b/multi_token_scarcity_setting", exist_ok=True)
df = pd.DataFrame(pandas_data)
df.to_csv("results/baseline_probes_gemma-2-2b/multi_token_scarcity_setting/all_metrics.csv", index=False)

# %%

