# Are Sparse Autoencoders Useful? A Case Study in Sparse Probing
<img width="1213" alt="Screenshot 2025-02-24 at 9 58 54 PM" src="https://github.com/user-attachments/assets/09a20f0b-9f45-4382-b6c2-e70bba6c17db" />

This repository contains code to replicate experiments from the paper *Are Sparse Autoencoders Useful? A Case Study in Sparse Probing*. The workflow of our code involves three primary stages:

1. **Generating Model and SAE Activations:**
   - Model and SAE activations for probing datasets are generated in `JOSH FILL IN`.
   - OOD regime activations are specifically generated in `plot_ood.ipynb`.

2. **Creating Probes:**
   - Baseline probes are created using `run_baselines.py`. This script also includes additional functions for OOD experiments related to probe pruning and latent interpretability (see Sections 4.1 and 4.2 of the paper).

3. **Visualizing Results:**
   - Standard condition plots: `plot_normal.ipynb`
   - Data scarcity, class imbalance, and corrupted data regimes: `plot_combined.ipynb`
   - OOD plots: `plot_ood.ipynb`
   - Llama-3.1-8B results replication: `plot_llama.ipynb`
   - GLUE CoLA and AIMade investigations (Sections 4.3.1 and 4.3.2): `dataset_investigations/`
   - SAE architectural improvements (Section 6): `sae_improvement.ipynb`

### Datasets
- **Raw Text Datasets:** Accessible via [Dropbox link](https://www.dropbox.com/scl/fo/lvajx9100jsy3h9cvis7q/AIocXXICIwHsz-HsXSekC3Y?rlkey=tq7td61h1fufm01cbdu2oqsb5&st=aorlnph5&dl=0).
- **Model Activations:** Also stored on Dropbox (Note: Files are large).

```

For any questions or clarifications, please open an issue or reach out to us!

```

