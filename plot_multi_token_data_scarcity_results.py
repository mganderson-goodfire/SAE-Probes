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
num_train = 103

sae_multi_token_results_limited_data = pkl.load(open(f"results/multi_token_sae_layer20_numtrain{num_train}.pkl", "rb"))
sae_multi_token_results = pkl.load(open(f"results/multi_token_sae_layer20_numtrainall.pkl", "rb"))

baseline_multi_token_results = pkl.load(open("results/multi_token_baseline_layer20_numtrainall.pkl", "rb"))
baseline_multi_token_results_limited_data = pkl.load(open(f"results/multi_token_baseline_layer20_numtrain{num_train}.pkl", "rb"))
# %%

baseline_test_aucs_single_token = baseline_single_token_results[baseline_single_token_results["method"] == "logreg"]
baseline_test_aucs_single_token = baseline_test_aucs_single_token[["dataset", "test_auc", "val_auc"]]

baseline_multi_token_test_auc = []
for row in baseline_multi_token_results:
    baseline_multi_token_test_auc.append({"dataset": row["dataset"], "test_auc": row["test_auc"], "val_auc": row["val_auc"]})
baseline_multi_token_test_auc = pd.DataFrame(baseline_multi_token_test_auc)

baseline_single_token_limited_data_test_auc = baseline_single_token_scarce_results[
    (baseline_single_token_scarce_results["method"] == "logreg") & 
    (baseline_single_token_scarce_results["num_train"] == num_train)]
baseline_single_token_limited_data_test_auc = baseline_single_token_limited_data_test_auc[["dataset", "test_auc", "val_auc"]]

baseline_multi_token_limited_data_test_auc = []
for row in baseline_multi_token_results_limited_data:
    baseline_multi_token_limited_data_test_auc.append({"dataset": row["dataset"], "test_auc": row["test_auc"], "val_auc": row["val_auc"]})
baseline_multi_token_limited_data_test_auc = pd.DataFrame(baseline_multi_token_limited_data_test_auc)


k = 128

sae_single_token_test_auc = sae_single_token_results[sae_single_token_results["sae_id"] == "layer_20/width_16k/average_l0_408"][sae_single_token_results["k"] == k]
sae_single_token_test_auc = sae_single_token_test_auc[["dataset", "test_auc", "val_auc"]]

sae_single_token_limited_data_test_auc = sae_single_token_scarce_results[sae_single_token_scarce_results["sae_id"] == "layer_20/width_16k/average_l0_408"][sae_single_token_scarce_results["k"] == k][sae_single_token_scarce_results["num_train"] == num_train]
sae_single_token_limited_data_test_auc = sae_single_token_limited_data_test_auc[["dataset", "test_auc", "val_auc"]]    

sae_multi_token_test_auc_limited_data = []
for row in sae_multi_token_results_limited_data:
    sae_multi_token_test_auc_limited_data.append({"dataset": row["dataset"], "test_auc": row["metrics"]["test_auc"], "val_auc": row["metrics"]["val_auc"]})
sae_multi_token_test_auc_limited_data = pd.DataFrame(sae_multi_token_test_auc_limited_data)

sae_multi_token_test_auc = []
for row in sae_multi_token_results:
    sae_multi_token_test_auc.append({"dataset": row["dataset"], "test_auc": row["metrics"]["test_auc"], "val_auc": row["metrics"]["val_auc"]})
sae_multi_token_test_auc = pd.DataFrame(sae_multi_token_test_auc)


# %%
plot_col = "test_auc"

# Create a matrix of test AUCs for each dataset and method
datasets = baseline_test_aucs_single_token["dataset"].values
# filtered_datasets = [d for d in datasets if int(d.split("_")[0]) not in range(65, 86)]
filtered_datasets = datasets
methods = ["Single Token (All Data)", "Single Token SAE (All Data)", "Multi Token Baseline (All Data)", "Multi Token SAE (All Data)",
           f"Single Token ({num_train} Examples)", f"Single Token SAE ({num_train} Examples)", f"Multi Token Baseline ({num_train} Examples)", f"Multi Token SAE ({num_train} Examples)"]

# Filter the datasets and corresponding AUCs
filtered_mask = [d in filtered_datasets for d in datasets]
filtered_auc_matrix = np.zeros((len(filtered_datasets), len(methods)))

# Fill in the matrix with filtered data
filtered_auc_matrix[:,0] = baseline_test_aucs_single_token[filtered_mask][plot_col].values
filtered_auc_matrix[:,1] = sae_single_token_test_auc[filtered_mask][plot_col].values
filtered_auc_matrix[:,2] = baseline_multi_token_test_auc[filtered_mask][plot_col].values
filtered_auc_matrix[:,3] = sae_multi_token_test_auc[filtered_mask][plot_col].values
filtered_auc_matrix[:,4] = baseline_single_token_limited_data_test_auc[filtered_mask][plot_col].values
filtered_auc_matrix[:,5] = sae_single_token_limited_data_test_auc[filtered_mask][plot_col].values
filtered_auc_matrix[:,6] = baseline_multi_token_limited_data_test_auc[filtered_mask][plot_col].values
filtered_auc_matrix[:,7] = sae_multi_token_test_auc_limited_data[filtered_mask][plot_col].values

# Create the heatmap
plt.figure(figsize=(12, 8))
im = plt.imshow(filtered_auc_matrix, aspect='auto')

# Add colorbar
plt.colorbar(im)

# Add labels
plt.xticks(np.arange(len(methods)), methods, rotation=45, ha='right')
plt.yticks(np.arange(len(filtered_datasets)), filtered_datasets)

# Add title
plt.title(f"{plot_col} by Dataset and Method")

# Adjust layout to prevent label cutoff
plt.tight_layout()

plt.show()

# Calculate filtered means
filtered_mean_aucs = {
    "Single Token (All Data)": baseline_test_aucs_single_token[filtered_mask][plot_col].mean(),
    "Single Token SAE (All Data)": sae_single_token_test_auc[filtered_mask][plot_col].mean(),
    "Multi Token Baseline (All Data)": baseline_multi_token_test_auc[filtered_mask][plot_col].mean(),
    "Multi Token SAE (All Data)": sae_multi_token_test_auc[filtered_mask][plot_col].mean(),
    f"Single Token ({num_train} Examples)": baseline_single_token_limited_data_test_auc[filtered_mask][plot_col].mean(),
    f"Single Token SAE ({num_train} Examples)": sae_single_token_limited_data_test_auc[filtered_mask][plot_col].mean(),
    f"Multi Token Baseline ({num_train} Examples)": baseline_multi_token_limited_data_test_auc[filtered_mask][plot_col].mean(),
    f"Multi Token SAE ({num_train} Examples)": sae_multi_token_test_auc_limited_data[filtered_mask][plot_col].mean()
}   

# Create bar plot of filtered averages
plt.figure(figsize=(10, 6))
plt.bar(range(len(filtered_mean_aucs)), filtered_mean_aucs.values())
plt.xticks(range(len(filtered_mean_aucs)), filtered_mean_aucs.keys(), rotation=45, ha='right')
plt.ylabel('Mean Test AUC')
plt.title(f'Average {plot_col} by Method')
plt.tight_layout()
plt.show()


# %%

# Sort all dataframes by dataset
baseline_single_token_limited_data_test_auc = baseline_single_token_limited_data_test_auc.sort_values("dataset")
sae_single_token_limited_data_test_auc = sae_single_token_limited_data_test_auc.sort_values("dataset")
baseline_multi_token_limited_data_test_auc = baseline_multi_token_limited_data_test_auc.sort_values("dataset")
sae_multi_token_test_auc_limited_data = sae_multi_token_test_auc_limited_data.sort_values("dataset")

# %%
# Create dataframe with results for each dataset
results_df = pd.DataFrame({
    "Dataset": filtered_datasets,
    f"Single Token ({num_train} Examples) Test": baseline_single_token_limited_data_test_auc[filtered_mask]["test_auc"].values,
    f"Single Token ({num_train} Examples) Val": baseline_single_token_limited_data_test_auc[filtered_mask]["val_auc"].values,
    f"Single Token SAE ({num_train} Examples) Test": sae_single_token_limited_data_test_auc[filtered_mask]["test_auc"].values,
    f"Single Token SAE ({num_train} Examples) Val": sae_single_token_limited_data_test_auc[filtered_mask]["val_auc"].values,
    f"Multi Token ({num_train} Examples) Test": baseline_multi_token_limited_data_test_auc[filtered_mask]["test_auc"].values,
    f"Multi Token ({num_train} Examples) Val": baseline_multi_token_limited_data_test_auc[filtered_mask]["val_auc"].values,
    f"Multi Token SAE ({num_train} Examples) Test": sae_multi_token_test_auc_limited_data[filtered_mask]["test_auc"].values,
    f"Multi Token SAE ({num_train} Examples) Val": sae_multi_token_test_auc_limited_data[filtered_mask]["val_auc"].values
})

# First analyze all methods including SAE
val_cols = [col for col in results_df.columns if "Val" in col]
test_cols = [col for col in results_df.columns if "Test" in col]

# For each dataset, find best method by validation and get corresponding test score
best_method_test_scores = []
for idx, row in results_df.iterrows():
    # Get validation scores for this dataset
    val_scores = row[val_cols]
    best_val_method = val_scores.idxmax()
    # Get corresponding test score
    test_method = best_val_method.replace("Val", "Test")
    best_method_test_scores.append(row[test_method])

print("\nAnalysis including SAE methods:")
print(f"Average test AUC when selecting best method per dataset by validation: {np.mean(best_method_test_scores):.3f}")

# Now analyze without SAE methods
non_sae_val_cols = [col for col in val_cols if "SAE" not in col]
non_sae_test_cols = [col for col in test_cols if "SAE" not in col]

# For each dataset, find best non-SAE method by validation and get corresponding test score
best_method_test_scores_no_sae = []
for idx, row in results_df.iterrows():
    # Get validation scores for this dataset
    val_scores = row[non_sae_val_cols]
    best_val_method = val_scores.idxmax()
    # Get corresponding test score
    test_method = best_val_method.replace("Val", "Test")
    best_method_test_scores_no_sae.append(row[test_method])

print("\nAnalysis excluding SAE methods:")
print(f"Average test AUC when selecting best method per dataset by validation: {np.mean(best_method_test_scores_no_sae):.3f}")

# %%


# Perform paired t-test to compare with SAE vs without SAE
from scipy import stats
t_stat, p_value = stats.ttest_rel(best_method_test_scores, best_method_test_scores_no_sae, alternative='greater')
print(f"\nPaired one-sided t-test (H1: With SAE > Without SAE):")
print(f"t-statistic: {t_stat:.6f}")
print(f"p-value: {p_value:.6f}")

# Create comparison matrix
comparison_matrix = np.vstack([best_method_test_scores, best_method_test_scores_no_sae])

# Create figure and axis
plt.figure(figsize=(15, 4))
im = plt.imshow(comparison_matrix, aspect='auto', vmin=0.5, vmax=1.0)

# Add colorbar
plt.colorbar(im)

# Set labels
plt.yticks([0, 1], ['With SAE', 'Without SAE'], fontsize=12)
plt.xticks(range(len(filtered_datasets)), filtered_datasets, rotation=45, ha='right', fontsize=5.5)

# Add title
plt.title(f'Test AUC For Quivers, {num_train} Training Examples\np value for hypothesis that w/ SAE is better: {p_value:.6f}\nAvg AUC w/ SAE: {np.mean(best_method_test_scores):.3f}, Avg AUC w/o SAE: {np.mean(best_method_test_scores_no_sae):.3f}')

# Adjust layout
plt.tight_layout()


# %%

# Sort all dataframes by dataset
baseline_test_aucs_single_token = baseline_test_aucs_single_token.sort_values("dataset")
sae_single_token_test_auc = sae_single_token_test_auc.sort_values("dataset") 
baseline_multi_token_test_auc = baseline_multi_token_test_auc.sort_values("dataset")
sae_multi_token_test_auc = sae_multi_token_test_auc.sort_values("dataset")


# %%
results_df = pd.DataFrame({
    "Dataset": filtered_datasets,
    f"Single Token (All Data) Test": baseline_test_aucs_single_token[filtered_mask]["test_auc"].values,
    f"Single Token (All Data) Val": baseline_test_aucs_single_token[filtered_mask]["val_auc"].values,
    f"Single Token SAE (All Data) Test": sae_single_token_test_auc[filtered_mask]["test_auc"].values,
    f"Single Token SAE (All Data) Val": sae_single_token_test_auc[filtered_mask]["val_auc"].values,
    f"Multi Token (All Data) Test": baseline_multi_token_test_auc[filtered_mask]["test_auc"].values,
    f"Multi Token (All Data) Val": baseline_multi_token_test_auc[filtered_mask]["val_auc"].values,
    f"Multi Token SAE (All Data) Test": sae_multi_token_test_auc[filtered_mask]["test_auc"].values,
    f"Multi Token SAE (All Data) Val": sae_multi_token_test_auc[filtered_mask]["val_auc"].values
})
    

# Now analyze full data results
val_cols = [col for col in results_df.columns if "Val" in col]
test_cols = [col for col in results_df.columns if "Test" in col]


# For each dataset, find best method by validation and get corresponding test score
best_method_test_scores_full = []
for idx, row in results_df.iterrows():
    # Get validation scores for this dataset
    val_scores = row[val_cols]
    best_val_method = val_scores.idxmax()
    # Get corresponding test score
    test_method = best_val_method.replace("Val", "Test")
    best_method_test_scores_full.append(row[test_method])

print("\nAnalysis for full data methods:")
print(f"Average test AUC when selecting best method per dataset by validation: {np.mean(best_method_test_scores_full):.3f}")

# Now analyze without SAE methods for full data
non_sae_full_val_cols = [col for col in val_cols if "SAE" not in col]
non_sae_full_test_cols = [col for col in test_cols if "SAE" not in col]

# For each dataset, find best non-SAE method by validation and get corresponding test score
best_method_test_scores_full_no_sae = []
for idx, row in results_df.iterrows():
    # Get validation scores for this dataset
    val_scores = row[non_sae_full_val_cols]
    best_val_method = val_scores.idxmax()
    # Get corresponding test score
    test_method = best_val_method.replace("Val", "Test")
    best_method_test_scores_full_no_sae.append(row[test_method])

print("\nAnalysis excluding SAE methods (full data):")
print(f"Average test AUC when selecting best method per dataset by validation: {np.mean(best_method_test_scores_full_no_sae):.3f}")

# Create comparison matrix
comparison_matrix_full = np.vstack([best_method_test_scores_full, best_method_test_scores_full_no_sae])


# Perform paired t-test to compare with SAE vs without SAE
t_stat_full, p_value_full = stats.ttest_rel(best_method_test_scores_full, best_method_test_scores_full_no_sae, alternative='greater')
print(f"\nPaired one-sided t-test for full data (H1: With SAE > Without SAE):")
print(f"t-statistic: {t_stat_full:.6f}")
print(f"p-value: {p_value_full:.6f}")

# Create figure and axis
plt.figure(figsize=(15, 4))
im = plt.imshow(comparison_matrix_full, aspect='auto', vmin=0.5, vmax=1.0)

# Add colorbar
plt.colorbar(im)

# Set labels
plt.yticks([0, 1], ['With SAE', 'Without SAE'], fontsize=12)
plt.xticks(range(len(filtered_datasets)), filtered_datasets, rotation=45, ha='right', fontsize=5.5)

# Add title
plt.title(f'Test AUC For Quivers, Full Data \n p value for hypothesis that w/ SAE is better: {p_value_full:.6f}\nAvg AUC w/ SAE: {np.mean(best_method_test_scores_full):.3f}, Avg AUC w/o SAE: {np.mean(best_method_test_scores_full_no_sae):.3f}')

# Adjust layout
plt.tight_layout()




# %%











