# %%

from natsort import natsorted
import pandas as pd

baseline_single_token_results = pd.read_csv("results/baseline_probes_gemma-2-9b/normal_settings/layer20_results.csv")
sae_single_token_results = pd.read_csv("results/sae_probes_gemma-2-9b/normal_setting/all_metrics.csv")
# %%

baseline_single_token_results = baseline_single_token_results[baseline_single_token_results["method"] == "logreg"]
sae_single_token_results = sae_single_token_results[sae_single_token_results["k"] == 128]

target_sae_id = "layer_20/width_16k/average_l0_408"
sae_single_token_results = sae_single_token_results[sae_single_token_results["sae_id"] == target_sae_id]

# %%


datasets = sae_single_token_results["dataset"].unique()

average_sae_test_auc = sae_single_token_results["test_auc"].mean()
average_baseline_test_auc = baseline_single_token_results["test_auc"].mean()

print(f"Average SAE Test AUC: {average_sae_test_auc}")
print(f"Average Baseline Test AUC: {average_baseline_test_auc}")

# %%
from natsort import natsorted
combined_test_aucs = []
for dataset in natsorted(datasets):
    sae_val_auc = sae_single_token_results[sae_single_token_results["dataset"] == dataset]["val_auc"].values[0]
    baseline_val_auc = baseline_single_token_results[baseline_single_token_results["dataset"] == dataset]["val_auc"].values[0]
    sae_test_auc = sae_single_token_results[sae_single_token_results["dataset"] == dataset]["test_auc"].values[0]
    baseline_test_auc = baseline_single_token_results[baseline_single_token_results["dataset"] == dataset]["test_auc"].values[0]
    if sae_val_auc > baseline_val_auc:
        combined_test_aucs.append(sae_test_auc)
    else:
        combined_test_aucs.append(baseline_test_auc)
    print(dataset, combined_test_aucs[-1], sae_test_auc, baseline_test_auc)

import numpy as np
print(f"Average Combined Test AUC: {np.mean(combined_test_aucs)}")

# %%

print(sae_single_token_results)

# %%
