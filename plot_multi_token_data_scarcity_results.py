# %%
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
# %%

num_trains = [100, 1024]
k = 128

baseline_single_token_results = pd.read_csv("results/baseline_probes_gemma-2-2b/scarcity/all_results.csv")
sae_single_token_results = pd.read_csv("results/sae_probes_gemma-2-2b/scarcity_setting/all_metrics.csv")
baseline_multi_token_results = pd.read_csv("results/baseline_probes_gemma-2-2b/multi_token_scarcity_setting/all_metrics.csv")
sae_multi_token_results = pd.read_csv("results/sae_probes_gemma-2-2b/multi_token_scarcity_setting/all_metrics.csv")

# Replace num_train > 100 with 1024 for baseline dfs
baseline_single_token_results.loc[baseline_single_token_results["num_train"] > 100, "num_train"] = 1024
baseline_multi_token_results.loc[baseline_multi_token_results["num_train"] > 100, "num_train"] = 1024

sae_id_to_friendly_name = {
    "gemma-2-2b_top_k_width-2pow14_date-0107/resid_post_layer_12/trainer_2": "TopK",
    "gemma-2-2b_p_anneal_width-2pow14_date-0107/resid_post_layer_12/trainer_4": "P Anneal",
    "gemma-2-2b_batch_top_k_width-2pow14_date-0107/resid_post_layer_12/trainer_8": "Batch TopK",
    "gemma-2-2b_standard_new_width-2pow14_date-0107/resid_post_layer_12/trainer_4": "Anthropic April Update",
    "gemma-2-2b_gated_width-2pow14_date-0107/resid_post_layer_12/trainer_4": "Gated",
    "gemma-2-2b_matryoshka_batch_top_k_width-2pow14_date-0107/resid_post_layer_12/trainer_2": "Matryoshka",
    "gemma-2-2b_jump_relu_width-2pow14_date-0107/resid_post_layer_12/trainer_2": "Jump ReLU",
    "old_relu_google_gemma-2-2b_standard/resid_post_layer_12/trainer_4": "Original ReLU"
}
sae_multi_token_results["sae_id"] = sae_multi_token_results["sae_id"].map(sae_id_to_friendly_name)
sae_single_token_results["sae_id"] = sae_single_token_results["sae_id"].map(sae_id_to_friendly_name)

# Filter out results for which k != 128
sae_single_token_results = sae_single_token_results[sae_single_token_results["k"] == k]

sae_order = ["Original ReLU", "Anthropic April Update", "Gated", "TopK", "Jump ReLU", "Batch TopK", "P Anneal", "Matryoshka"]
# %%

# Get unique datasets from each dataframe
baseline_single_datasets = set(baseline_single_token_results["dataset"].unique())
sae_single_datasets = set(sae_single_token_results["dataset"].unique())
baseline_multi_datasets = set(baseline_multi_token_results["dataset"].unique())
sae_multi_datasets = set(sae_multi_token_results["dataset"].unique())

# Find intersection of all datasets
common_datasets = baseline_single_datasets.intersection(sae_single_datasets, baseline_multi_datasets, sae_multi_datasets)
print("Common datasets across all dataframes:", len(common_datasets))

unique_sae_ids = sae_multi_token_results["sae_id"].unique()

# %%

# Use common datasets
filtered_datasets = list(common_datasets)

# Create list of methods including each unique SAE ID
methods = []
for num_train in num_trains:
    methods.extend([
        f"Single Token Baseline ({num_train} Examples)",
        f"Multi Token Baseline ({num_train} Examples)"
    ])
    for sae_id in unique_sae_ids:
        methods.extend([
            f"Single Token {sae_id} ({num_train} Examples)",
            f"Multi Token {sae_id} ({num_train} Examples)"
        ])

# %%
plot_col = "test_auc"

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

for i, num_train in enumerate(num_trains):
    ax = ax1 if i == 0 else ax2
    
    # List to store methods and their averages for this num_train
    method_averages = []
    simple_labels = []
    
    # Add baseline single token results first
    filtered_baseline_single = baseline_single_token_results[
        baseline_single_token_results["num_train"] == num_train
    ]
    method_averages.append(filtered_baseline_single[plot_col].mean())
    simple_labels.append("Single Token Baseline")

    # Add single token SAE results
    for sae_id in sae_order:
        filtered_sae_single = sae_single_token_results[
            (sae_single_token_results["num_train"] == num_train) &
            (sae_single_token_results["sae_id"] == sae_id)
        ]
        if len(filtered_sae_single) > 0:
            method_averages.append(filtered_sae_single[plot_col].mean())
            simple_labels.append(f"Single Token {sae_id}")
        else:
            print(f"No single token SAE results for {num_train} examples and {sae_id}")

    # Add baseline multi token results
    filtered_baseline_multi = baseline_multi_token_results[
        baseline_multi_token_results["num_train"] == num_train
    ]
    method_averages.append(filtered_baseline_multi[plot_col].mean())
    simple_labels.append("Multi Token Baseline")

    # Add multi token SAE results
    for sae_id in sae_order:
        filtered_sae_multi = sae_multi_token_results[
            (sae_multi_token_results["num_train"] == num_train) &
            (sae_multi_token_results["sae_id"] == sae_id)
        ]
        method_averages.append(filtered_sae_multi[plot_col].mean())
        simple_labels.append(f"Multi Token {sae_id}")
    
    # Plot the data
    bars = ax.bar(range(len(simple_labels)), method_averages)
    ax.set_xticks(range(len(simple_labels)))
    ax.set_xticklabels(simple_labels, rotation=45, ha='right')
    ax.set_ylabel('Average Test AUC')
    ax.set_title(f'Average Test AUC by Method ({num_train} Training Examples)')
    
    # Add vertical line between single and multi token results
    single_token_count = 1 + sum(1 for sae_id in sae_order if len(sae_single_token_results[
        (sae_single_token_results["num_train"] == num_train) &
        (sae_single_token_results["sae_id"] == sae_id)
    ]) > 0)
    ax.axvline(x=single_token_count-0.5, color='red', linestyle=':', alpha=1, linewidth=2)

    ax.set_ylim(0.75, 1.0)

plt.tight_layout()



# %%

# %%
import pickle
old_sae_results = pickle.load(open("results/multi_token_sae_layer20_numtrain103.pkl", "rb"))
old_dataset_to_test_auc_and_val_auc = {}
for row in old_sae_results:
    old_dataset_to_test_auc_and_val_auc[row["dataset"]] = (row["metrics"]["test_auc"], row["metrics"]["val_auc"])

# %%

# Create dataframe with results for each dataset and SAE ID
from natsort import natsorted

id_and_num_train_to_results = {}
for num_train in sorted(num_trains):

    for sae_id in sae_order + ["Old Results", "No SAE"]:
        results = {} # Dataset -> Test AUC from quiver

        for dataset in filtered_datasets:
            val_aucs = []
            test_aucs = []

            res = baseline_single_token_results[
                (baseline_single_token_results["num_train"] == num_train) &
                (baseline_single_token_results["dataset"] == dataset)
            ]
            val_aucs.append(res["val_auc"].iloc[0])
            test_aucs.append(res["test_auc"].iloc[0])

            if sae_id not in ["Old Results", "No SAE"]:
                res = sae_single_token_results[
                    (sae_single_token_results["num_train"] == num_train) &
                    (sae_single_token_results["dataset"] == dataset) &
                    (sae_single_token_results["sae_id"] == sae_id)
                ]
                # TODO: Fix this once finished
                assert len(res) <= 1
                if len(res) == 1:
                    val_aucs.append(res["val_auc"].iloc[0])
                    test_aucs.append(res["test_auc"].iloc[0])

            res = baseline_multi_token_results[
                (baseline_multi_token_results["num_train"] == num_train) &
                (baseline_multi_token_results["dataset"] == dataset)
            ]
            val_aucs.append(res["val_auc"].iloc[0])
            test_aucs.append(res["test_auc"].iloc[0])

            if sae_id not in ["Old Results", "No SAE"]:
                res = sae_multi_token_results[
                    (sae_multi_token_results["num_train"] == num_train) &
                    (sae_multi_token_results["dataset"] == dataset) &
                    (sae_multi_token_results["sae_id"] == sae_id)
                ]
                val_aucs.append(res["val_auc"].iloc[0])
                test_aucs.append(res["test_auc"].iloc[0])

            if sae_id == "Old Results":
                test_aucs.append(old_dataset_to_test_auc_and_val_auc[dataset][0])
                val_aucs.append(old_dataset_to_test_auc_and_val_auc[dataset][1])
                
            best_val_auc_index = np.argmax(val_aucs)
            test_auc = test_aucs[best_val_auc_index]
            results[dataset] = test_auc

        id_and_num_train_to_results[(sae_id, num_train)] = results



# Calculate average results for each SAE ID and num_train combination
for num_train in sorted(num_trains):
    # Create matrix of results - one row per SAE ID, including baseline
    results_matrix = []
    labels = []
    
    
    # Get results for each SAE
    sae_and_mean_performance = []
    for sae_id in ["No SAE"] + sae_order + ["Old Results"]:
        sae_results = []
        labels.append(f'{sae_id}')
        
        results_dict = id_and_num_train_to_results[(sae_id, num_train)]
        for dataset in natsorted(filtered_datasets):
            sae_results.append(results_dict[dataset])

        sae_and_mean_performance.append((sae_id, np.mean(sae_results)))
        print(sae_and_mean_performance[-1])
        results_matrix.append(sae_results)

    # Convert to numpy array
    results_matrix = np.array(results_matrix)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 0.75 * len(labels)), 
                                  gridspec_kw={'width_ratios': [3, 1]})
    # Use reversed Reds colormap
    cmap = plt.cm.Reds_r
    norm = plt.Normalize(vmin=0.5, vmax=1.0)

    # Heatmap plot
    im = ax1.imshow(results_matrix, aspect='auto', cmap=cmap, norm=norm)
    plt.colorbar(im, ax=ax1)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=12)
    ax1.set_xticks(range(len(filtered_datasets)))
    ax1.set_xticklabels(natsorted(filtered_datasets), rotation=45, ha='right', fontsize=5.5)
    ax1.set_title(f'Test AUC Per Dataset For Quivers, {num_train} Training Examples')

    # Bar plot of averages
    ax2.barh(range(len(labels)), [x[1] for x in sae_and_mean_performance][::-1])
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels([])  # Hide labels since they're shown in heatmap
    ax2.set_xlabel('Average Test AUC')
    ax2.set_title('Average Test AUCs For Quivers')
    ax2.set_xlim(0.7, 1.0)  # Set x-axis limits to start at 0.7
    
    # Add text labels on bars
    for i, score in enumerate([x[1] for x in sae_and_mean_performance][::-1]):
        ax2.text(score, i, f'{score:.3f}', va='center')

    plt.tight_layout()
    plt.show()




# %%


# Load and process data
sae_df = pd.read_csv("results/sae_probes_gemma-2-2b/multi_token_scarcity_setting/all_metrics.csv")
sae_df["sae_id"] = sae_df["sae_id"].map(sae_id_to_friendly_name)

sae_id_and_dataset_to_test_auc = {}
for sae_id in sae_order:
    for dataset in sae_df["dataset"].unique():
        res = sae_df[
            (sae_df["sae_id"] == sae_id) &
            (sae_df["dataset"] == dataset) &
            (sae_df["num_train"] == 100)
        ]
        if len(res) > 0:
            sae_id_and_dataset_to_test_auc[(sae_id, dataset)] = res["test_auc"].iloc[0]


# Convert dictionary to matrix form
datasets = natsorted(sae_df["dataset"].unique())
sae_ids = sae_order + ["Old Results"]  # Add old results label

results_matrix = np.zeros((len(sae_ids), len(datasets)))
results_matrix[:] = np.nan

# Fill in SAE results
for i, sae_id in enumerate(sae_order):
    for j, dataset in enumerate(datasets):
        if (sae_id, dataset) in sae_id_and_dataset_to_test_auc:
            results_matrix[i,j] = sae_id_and_dataset_to_test_auc[(sae_id, dataset)]

# Fill in old results
for j, dataset in enumerate(datasets):
    if dataset in old_dataset_to_test_auc:
        results_matrix[-1,j] = old_dataset_to_test_auc[dataset]

# Plot heatmap
plt.figure(figsize=(15, 8))
im = plt.imshow(results_matrix, aspect='auto', cmap='Reds_r', norm=plt.Normalize(vmin=0.5, vmax=1.0))
plt.colorbar(im)

plt.yticks(range(len(sae_ids)), sae_ids, fontsize=12)
plt.xticks(range(len(datasets)), datasets, rotation=45, ha='right', fontsize=5.5)
plt.title('Test AUC Per Dataset For Each SAE')

plt.tight_layout()
plt.show()

# %%


