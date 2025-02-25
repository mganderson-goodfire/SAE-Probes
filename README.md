# Are Sparse Autoencoders Useful? A Case Study in Sparse Probing
<img width="1213" alt="Screenshot 2025-02-24 at 9 58 54 PM" src="https://github.com/user-attachments/assets/09a20f0b-9f45-4382-b6c2-e70bba6c17db" />

This repository contains code to replicate experiments from the paper *Are Sparse Autoencoders Useful? A Case Study in Sparse Probing*. The workflow of our code involves three primary stages:

1. **Generating Model and SAE Activations:**
   - Model activations for probing datasets are generated in `generate_model_activations.py`
   - SAE activations are generated in `generate_sae_activations.py`
   - OOD regime activations are specifically generated in `plot_ood.ipynb`.
   - Mutli-token activations are specifically generated in `generate_model_and_sae_multi_token_acts.py`. Caution: this will take up a lot of memory (~1TB).

2. **Training Probes:**
   - Baseline probes are trained using `run_baselines.py`. This script also includes additional functions for OOD experiments related to probe pruning and latent interpretability (see Sections 4.1 and 4.2 of the paper).
   - SAE probes are trained using `train_sae_probes.py`. Sklearn regression is most efficient when run in a single thread, and then many of those threads can be run in parallel. We include an example of how to do this in `train_sae_probes.sh`.
   - Multi token SAE probes and baseline probes are trained using `run_multi_token_acts.py`.
   - Combining all results into csvs after they are done is done with `combine_results.py`.

3. **Visualizing Results:**
   - Standard condition plots: `plot_normal.ipynb`
   - Data scarcity, class imbalance, and corrupted data regimes: `plot_combined.ipynb`
   - OOD plots: `plot_ood.ipynb`
   - Llama-3.1-8B results replication: `plot_llama.ipynb`
   - GLUE CoLA and AIMade investigations (Sections 4.3.1 and 4.3.2): `dataset_investigations/`
   - SAE architectural improvements (Section 6): `sae_improvement.ipynb`
   - Multi token: `plot_multi_token.py`

### Datasets
- **Raw Text Datasets:** Accessible via [Dropbox link](https://www.dropbox.com/scl/fo/lvajx9100jsy3h9cvis7q/AIocXXICIwHsz-HsXSekC3Y?rlkey=tq7td61h1fufm01cbdu2oqsb5&st=aorlnph5&dl=0).
- **Model Activations:** Also stored on Dropbox (Note: Files are large).

## Requirements
The required python packages to run this repo are
torch transformer_lens scikit-learn sae_lens sae_bench
We recommend you create a new python venv named probing and install these packages with pip:

```
python -m venv probing
source probing/bin/activate
pip install transformer_lens sae_lens transformers datasets torch adjustText circuitsvis ipython
```
Let us know if anything does not work with this environment!


For any questions or clarifications, please open an issue or reach out to us!

```

