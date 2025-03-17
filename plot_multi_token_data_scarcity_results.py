# %%
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
num_train = 20

baseline_single_token_results = pd.read_csv("results/baseline_probes_gemma-2-9b/normal_settings/layer20_results.csv")
baseline_single_token_scarce_results = pd.read_csv("results/baseline_probes_gemma-2-9b/scarcity/summary/all_results.csv")

sae_single_token_results = pd.read_csv("results/sae_probes_gemma-2-9b/normal_setting/all_metrics.csv")
sae_single_token_scarce_results = pd.read_csv("results/sae_probes_gemma-2-9b/scarcity_setting/all_metrics.csv")


# %%

sae_multi_token_results_numtrain20 = pkl.load(open("results/multi_token_sae_layer20_numtrain20.pkl", "rb"))

# %%

num_train = 20
baseline_test_aucs_single_token = baseline_single_token_results[baseline_single_token_results["method"] == "logreg"]
baseline_test_aucs_single_token = baseline_test_aucs_single_token[["dataset", "test_auc", "val_auc"]]

dataset_to_test_auc = []
for row in sae_multi_token_results_numtrain20:
    dataset_to_test_auc.append({"dataset": row["dataset"], "test_auc": row["metrics"]["test_auc"], "val_auc": row["metrics"]["val_auc"]})
dataset_to_test_auc = pd.DataFrame(dataset_to_test_auc)

baseline_test_aucs_single_token_scarce = baseline_single_token_scarce_results[
    (baseline_single_token_scarce_results["method"] == "logreg") & 
    (baseline_single_token_scarce_results["num_train"] == num_train)]
baseline_test_aucs_single_token_scarce = baseline_test_aucs_single_token_scarce[["dataset", "test_auc", "val_auc"]]

k = 128
sae_single_token_results_numtrain20 = sae_single_token_results[sae_single_token_results["sae_id"] == "layer_20/width_16k/average_l0_408"][sae_single_token_results["k"] == k]
sae_single_token_results_numtrain20 = sae_single_token_results_numtrain20[["dataset", "test_auc", "val_auc"]]

sae_single_token_scarce_results_numtrain20 = sae_single_token_scarce_results[sae_single_token_scarce_results["sae_id"] == "layer_20/width_16k/average_l0_408"][sae_single_token_scarce_results["k"] == k][sae_single_token_scarce_results["num_train"] == num_train]
sae_single_token_scarce_results_numtrain20 = sae_single_token_scarce_results_numtrain20[["dataset", "test_auc", "val_auc"]]    


# %%


plot_col = "val_auc"

# Create a matrix of test AUCs for each dataset and method
datasets = baseline_test_aucs_single_token["dataset"].values
filtered_datasets = [d for d in datasets if int(d.split("_")[0]) not in range(65, 86)]
methods = ["Single Token (All Data)", "Single Token (20 Examples)", 
           "Single Token SAE (All Data)", "Single Token SAE (20 Examples)", 
           "Multi Token SAE (20 Examples)"]

# Filter the datasets and corresponding AUCs
filtered_mask = [d in filtered_datasets for d in datasets]

filtered_auc_matrix = np.zeros((len(filtered_datasets), len(methods)))

# Fill in the matrix with filtered data
filtered_auc_matrix[:,0] = baseline_test_aucs_single_token[filtered_mask][plot_col].values
filtered_auc_matrix[:,1] = baseline_test_aucs_single_token_scarce[filtered_mask][plot_col].values 
filtered_auc_matrix[:,2] = sae_single_token_results_numtrain20[filtered_mask][plot_col].values
filtered_auc_matrix[:,3] = sae_single_token_scarce_results_numtrain20[filtered_mask][plot_col].values
filtered_auc_matrix[:,4] = dataset_to_test_auc[filtered_mask][plot_col].values

# Create the heatmap
plt.figure(figsize=(12, 8))
im = plt.imshow(filtered_auc_matrix, aspect='auto')

# Add colorbar
plt.colorbar(im)

# Add labels
plt.xticks(np.arange(len(methods)), methods, rotation=45, ha='right')
plt.yticks(np.arange(len(filtered_datasets)), filtered_datasets)

# Add title
plt.title(f"{plot_col} by Dataset and Method (Filtered)")

# Adjust layout to prevent label cutoff
plt.tight_layout()

plt.show()

# Calculate filtered means
filtered_mean_aucs = {
    "Single Token (All Data)": baseline_test_aucs_single_token[filtered_mask][plot_col].mean(),
    "Single Token (20 Examples)": baseline_test_aucs_single_token_scarce[filtered_mask][plot_col].mean(),
    "Single Token SAE (All Data)": sae_single_token_results_numtrain20[filtered_mask][plot_col].mean(),
    "Single Token SAE (20 Examples)": sae_single_token_scarce_results_numtrain20[filtered_mask][plot_col].mean(),
    "Multi Token SAE (20 Examples)": dataset_to_test_auc[filtered_mask][plot_col].mean()
}   

# Create bar plot of filtered averages
plt.figure(figsize=(10, 6))
plt.bar(range(len(filtered_mean_aucs)), filtered_mean_aucs.values())
plt.xticks(range(len(filtered_mean_aucs)), filtered_mean_aucs.keys(), rotation=45, ha='right')
plt.ylabel('Mean Test AUC')
plt.title(f'Average {plot_col} by Method (Filtered)')
plt.tight_layout()
plt.show()



# %%

print(datasets)




# %%











