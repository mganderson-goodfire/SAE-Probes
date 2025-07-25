# SAE-Probes Makefile
# Organized workflow for reproducing "Are Sparse Autoencoders Useful?" paper

# Python command using uv
PYTHON := uv run python
JUPYTER := uv run jupyter

# Default models and settings
MODEL ?= gemma-2-9b
SETTING ?= normal
LAYER ?= 20
DATASET ?= 100_news_fake

# Directories
DATA_DIR := raw_text_datasets
RESULTS_DIR := results
PLOTS_DIR := plots

.PHONY: help all clean setup
.DEFAULT_GOAL := help

# ==================== HELP ====================
help: ## Show this help message
	@echo "SAE-Probes Reproduction Makefile"
	@echo "================================"
	@echo ""
	@echo "Usage: make [target] [MODEL=gemma-2-9b] [SETTING=normal]"
	@echo ""
	@echo "Main Workflow Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Models: gemma-2-9b, llama-3.1-8b, gemma-2-2b"
	@echo "Settings: normal, scarcity_10, scarcity_100, class_imbalance, corrupt"

# ==================== SETUP ====================
setup: ## Initial setup: extract pre-computed data
	@echo "Extracting pre-computed data..."
	cd $(DATA_DIR) && tar -xzf model_activations_$(MODEL).tar.gz
	cd $(DATA_DIR) && tar -xzf model_activations_$(MODEL)_OOD.tar.gz
	cd $(DATA_DIR) && tar -xzf sae_activations_$(MODEL)_OOD.tar.gz
	cd $(DATA_DIR) && tar -xzf baseline_results_$(MODEL).tar.gz
	@echo "Setup complete!"

# ==================== ACTIVATION GENERATION (GPU REQUIRED) ====================
.PHONY: generate-activations generate-model-acts generate-sae-acts generate-sae-acts-all generate-sae-acts-optimized

generate-activations: generate-model-acts generate-sae-acts ## Generate all activations (GPU required)

generate-model-acts: ## Generate model activations for all datasets
	@echo "Generating model activations for $(MODEL)..."
	$(PYTHON) generate_model_activations.py --model $(MODEL)

generate-sae-acts: ## Generate SAE activations for specific setting
	@echo "Generating SAE activations for $(MODEL) with setting $(SETTING)..."
	$(PYTHON) generate_sae_activations.py --model $(MODEL) --setting $(SETTING) --device mps --model_name gemma-2-9b

generate-sae-acts-optimized: ## Run optimized SAE generation (much faster)
	@echo "Running optimized SAE generation pipeline..."
	bash save_sae_acts_optimized.sh

# Multi-token generation (separate due to high memory requirements)
generate-multi-token: ## Generate multi-token activations (high memory)
	@echo "Generating multi-token activations..."
	$(PYTHON) generate_model_and_sae_multi_token_acts.py

# ==================== PROBE TRAINING (CPU FRIENDLY) ====================
.PHONY: train-probes train-baselines train-sae-probes train-all-sae-probes train-all-probes

train-probes: train-baselines train-all-sae-probes ## Train all probes

train-baselines: ## Train baseline probes (logreg, knn, xgboost, etc.)
	@echo "Training baseline probes for $(MODEL) with setting $(SETTING)..."
	$(PYTHON) run_baselines.py --model $(MODEL) --setting $(SETTING)

train-sae-probes: ## Train SAE probes for specific setting
	@echo "Training SAE probes for $(MODEL) with setting $(SETTING)..."
	$(PYTHON) train_sae_probes.py --model_name $(MODEL) --setting $(SETTING) --reg_type l1

train-all-sae-probes: ## Train SAE probes for all experimental settings
	@echo "Training SAE probes for all settings..."
	$(MAKE) train-sae-probes SETTING=normal
	$(MAKE) train-sae-probes SETTING=scarcity
	$(MAKE) train-sae-probes SETTING=class_imbalance
	$(MAKE) train-sae-probes SETTING=OOD

train-multi-token: ## Train multi-token probes
	@echo "Training multi-token probes..."
	$(PYTHON) run_multi_token_acts.py


# ==================== RESULTS ANALYSIS ====================
.PHONY: combine-results analyze

combine-results: ## Combine all probe results into CSVs
	@echo "Combining results..."
	$(PYTHON) combine_results.py

analyze: combine-results ## Run full analysis pipeline

# ==================== VISUALIZATION ====================
.PHONY: visualize plot-all plot-normal plot-combined plot-ood plot-llama

visualize: plot-all ## Create all visualizations

plot-all: plot-normal plot-combined plot-ood ## Generate all main plots

plot-normal: ## Generate standard condition plots
	@echo "Generating normal condition plots..."
	$(JUPYTER) nbconvert --execute --to notebook --inplace plot_normal.ipynb

plot-combined: ## Generate combined experiment plots
	@echo "Generating combined experiment plots..."
	$(JUPYTER) nbconvert --execute --to notebook --inplace plot_combined.ipynb

plot-ood: ## Generate OOD experiment plots
	@echo "Generating OOD plots..."
	$(JUPYTER) nbconvert --execute --to notebook --inplace plot_ood.ipynb

plot-llama: ## Generate Llama-specific plots
	@echo "Generating Llama plots..."
	$(JUPYTER) nbconvert --execute --to notebook --inplace plot_llama.ipynb

plot-improvements: ## Generate SAE improvement analysis
	@echo "Generating SAE improvement plots..."
	$(JUPYTER) nbconvert --execute --to notebook --inplace sae_improvement.ipynb

# Standalone plot scripts
plot-k-vs-auc: ## Generate k vs AUC plots
	$(PYTHON) k_vs_auc_plot.py

plot-ai-human: ## Generate AI vs human text plots
	$(PYTHON) ai_vs_humanmade_plot.py

plot-multi-token-results: ## Generate multi-token plots
	$(PYTHON) plot_multi_token.py

plot-rebuttal: ## Generate rebuttal plots
	$(PYTHON) rebuttal_plots.py

# ==================== QUICK REPRODUCTION PATHS ====================
.PHONY: reproduce-from-scratch reproduce-from-precomputed reproduce-minimal

reproduce-from-scratch: ## Full reproduction from scratch (GPU required)
	@echo "Starting full reproduction pipeline..."
	$(MAKE) generate-activations
	$(MAKE) train-probes
	$(MAKE) analyze
	$(MAKE) visualize

reproduce-from-precomputed: ## Reproduce using pre-computed activations
	@echo "Reproducing from pre-computed data..."
	$(MAKE) setup
	$(MAKE) train-probes
	$(MAKE) analyze
	$(MAKE) visualize

reproduce-minimal: ## Minimal reproduction (just visualizations)
	@echo "Running minimal reproduction..."
	$(MAKE) setup
	$(MAKE) visualize

# ==================== UTILITIES ====================
.PHONY: clean clean-results clean-cache test-setup

clean: ## Clean all generated files
	rm -rf $(RESULTS_DIR)/*
	rm -rf $(PLOTS_DIR)/*
	rm -f *.pyc __pycache__/

clean-results: ## Clean only results (keep activations)
	rm -rf $(RESULTS_DIR)/*

clean-cache: ## Clean Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test-setup: ## Test if environment is properly configured
	@echo "Testing Python environment..."
	@$(PYTHON) -c "import torch; print('PyTorch:', torch.__version__)"
	@$(PYTHON) -c "import transformers; print('Transformers:', transformers.__version__)"
	@$(PYTHON) -c "import sae_lens; print('SAE Lens: OK')"
	@echo "Environment test passed!"

# ==================== DATASET INVESTIGATION ====================
.PHONY: investigate-cola investigate-aimade

investigate-cola: ## Run GLUE CoLA investigation
	$(JUPYTER) nbconvert --execute --to notebook --inplace dataset_investigations/gluecola_investigate.ipynb

investigate-aimade: ## Run AI-made text investigation
	$(JUPYTER) nbconvert --execute --to notebook --inplace dataset_investigations/aimade_investigation.ipynb

# ==================== ADVANCED OPTIONS ====================
# Run specific probe method
train-baseline-method: ## Train specific baseline method (use METHOD=logreg)
	$(PYTHON) run_baselines.py --model $(MODEL) --setting $(SETTING) --method $(METHOD)

# Run specific layer
train-layer: ## Train probes for specific layer (use LAYER=20)
	$(PYTHON) run_baselines.py --model $(MODEL) --layer $(LAYER)


# ==================== MONITORING ====================
.PHONY: check-gpu watch-training

check-gpu: ## Check GPU availability and memory
	@$(PYTHON) -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()) if torch.cuda.is_available() else None"

watch-training: ## Monitor training progress
	@watch -n 5 'ls -la results/*.csv | tail -20'