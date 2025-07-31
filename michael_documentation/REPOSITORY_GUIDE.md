# SAE-Probes Repository Guide

This document provides a comprehensive overview of the repository structure, explaining how each component relates to the overall reproduction workflow for "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing".

## Repository Structure Overview

The repository is organized into three main logical sections:

1. **Activation Generation**: Scripts that extract activations from models and SAEs (GPU-intensive)
2. **Probe Training**: Scripts that train various probes on the generated activations (CPU-friendly)
3. **Results Analysis & Visualization**: Tools for aggregating results and creating paper figures

## Section 1: Activation Generation (GPU Required)

### Model Activation Generation

#### `generate_model_activations.py`
- **Purpose**: Extracts intermediate layer activations from transformer models
- **Models Supported**: 
  - Gemma-2-9b (layers 20, 31, 41)
  - Llama-3.1-8b (layers 15, 23, 31)
  - Gemma-2-2b (layers 12, 18, 25)
- **Process**:
  1. Loads pre-trained transformer models via HookedTransformer
  2. Processes each dataset through the model
  3. Extracts activations at specified layers
  4. Saves as PyTorch tensors (.pt files)
- **Output Format**: `{dataset_id}_blocks.{layer}.hook_resid_post.pt`
- **Requirements**: CUDA GPU with 18GB+ VRAM for 9B models
- **Runtime**: Several hours depending on dataset size

### SAE Activation Generation

#### `generate_sae_activations.py`
- **Purpose**: Encodes model activations through Sparse Autoencoders
- **SAE Configurations**:
  - Width variants: 16k, 131k, 1M
  - Training variants: Different sparsity levels
- **Experimental Settings Supported**:
  - `normal`: Standard conditions
  - `scarcity_10/100`: Limited training samples
  - `class_imbalance`: 90% class skew
  - `corrupt`: 20% label noise
- **Process**:
  1. Loads pre-computed model activations
  2. Encodes through SAE
  3. Saves sparse representations
- **Output**: Sparse tensor files (compressed)
- **Requirements**: CUDA GPU
- **Memory Optimization**: Script restarts for each SAE to prevent memory leaks

#### `generate_model_and_sae_multi_token_acts.py`
- **Purpose**: Specialized generation for sequence-level analysis
- **Key Difference**: Preserves **all token positions** (not just last token)
- **Sequence Length**: 256 tokens (vs 1024 for standard generation)
- **Output Format**: `[batch_size, seq_len, hidden_dim]` instead of `[batch_size, hidden_dim]`
- **Use Case**: Tests if information is distributed across the sequence
- **Requirements**: Very high memory (~1TB mentioned in README)
- **Tokenization**: Left truncation, right padding to handle variable lengths

### Orchestration Script

#### `save_sae_acts_and_train_probes.sh`
- **Purpose**: Automates the full pipeline
- **Process**:
  1. Loops through each SAE configuration
  2. Generates SAE activations
  3. Immediately trains probes (20 parallel processes)
  4. Handles memory management between runs
- **Usage**: `bash save_sae_acts_and_train_probes.sh`

## Section 2: Probe Training

### Core Training Scripts

#### `run_baselines.py`
- **Purpose**: Trains baseline probes directly on transformer model activations
- **Models Supported**: Logistic Regression, PCA, KNN, XGBoost, MLP
- **Input**: Pre-computed model activations from `generate_model_activations.py`
- **Output**: Individual CSV files with test AUC scores
- **Key Features**:
  - Supports all experimental settings (normal, scarcity, imbalance, noise)
  - Cross-validation for hyperparameter tuning
  - Can run specific dataset/model/probe combinations

#### `train_sae_probes.py`
- **Purpose**: Trains L1/L2 regularized linear probes on SAE features
- **Input**: Pre-computed SAE activations from `generate_sae_activations.py`
- **Output**: Pickle files with probe results for different k values (1,2,4,...,512)
- **Key Features**:
  - Feature selection based on class differences
  - Tests different numbers of features (k)
  - Supports parallel execution via randomization

#### `run_multi_token_acts.py`
- **Purpose**: Specialized training for multi-token sequence analysis
- **Input**: Multi-token activations from `generate_model_and_sae_multi_token_acts.py`
- **Output**: Results for different aggregation methods (max, mean, attention)
- **Key Difference**: Uses **all token positions** instead of just the last token
- **Three Approaches**:
  1. **Aggregated SAE Probing**: Mean/max pooling across sequence positions
  2. **Baseline Concatenation**: PCA on each position + concatenate features
  3. **Attention Probing**: Learned attention weights across positions
- **Rationale**: Tests if information is distributed across multiple tokens rather than concentrated in the final token


## Section 3: Results Analysis & Visualization

### Results Aggregation

#### `combine_results.py`
- **Purpose**: Aggregates individual probe results into consolidated CSVs
- **Input**: Pickle files from probe training scripts
- **Process**: **No training** - pure data aggregation and flattening
- **Output**: Organized CSV files by model and setting
- **Usage**: Run after all probe training is complete

**Data Structure Details**:
- **Pickle Input**: Each `.pkl` file contains a list of ~10 dictionaries (one per k value)
- **Dictionary Format**:
  ```python
  {
      'test_f1': 0.85, 'test_acc': 0.87, 'test_auc': 0.92, 'val_auc': 0.89,
      'k': 1,  # Number of top SAE features used
      'dataset': '100_news_fake', 'layer': 20,
      'sae_id': 'layer_20/width_131k/average_l0_276',
      'reg_type': 'l1', 'binarize': False
  }
  ```
- **CSV Output**: One row per dictionary, ready for visualization
- **File Location**: `results/sae_probes_{model_name}/{setting}_setting/all_metrics.csv`

### Visualization

#### Main Visualization Notebooks

#### `plot_combined.ipynb`
- **Purpose**: Creates comprehensive comparison figures
- **Outputs**: Main paper figures showing SAE vs baseline performance
- **Covers**: All experimental settings in one view

#### `plot_normal.ipynb`
- **Purpose**: Standard condition visualizations
- **Focus**: Clean comparisons without data challenges

#### `plot_ood.ipynb`
- **Purpose**: Out-of-distribution experiment analysis
- **Special Features**: Distribution shift visualizations

#### `plot_llama.ipynb`
- **Purpose**: Llama-3.1-8b specific results
- **Use Case**: Model comparison analysis

#### `sae_improvement.ipynb`
- **Purpose**: Analyzes when SAEs outperform baselines
- **Output**: Section 6 figures on SAE architectural improvements

#### Standalone Visualization Scripts

#### `k_vs_auc_plot.py`
- **Purpose**: Creates grid plots of test AUC vs number of features
- **Output**: Per-dataset performance curves

#### `ai_vs_humanmade_plot.py`
- **Purpose**: Specialized analysis for AI detection task
- **Features**: Token distribution analysis, top feature visualization

#### `plot_multi_token.py`
- **Purpose**: Multi-token experiment visualizations
- **Compares**: Different aggregation strategies

#### `rebuttal_plots.py`
- **Purpose**: Additional plots for paper rebuttal
- **Includes**: Layer comparisons, before/after analyses

### Utility Modules

#### `utils_data.py`
- **Purpose**: Core data loading and preprocessing
- **Key Functions**:
  - `get_xyvals()`: Load activations and labels
  - `get_train_test_indices()`: Create splits with class balance control
  - `corrupt_labels()`: Add label noise for robustness testing
  - Dataset path management

#### `utils_sae.py`
- **Purpose**: SAE model management
- **Key Functions**:
  - SAE loading for different architectures
  - SAE ID mapping and organization
  - Activation encoding utilities

#### `utils_training.py`
- **Purpose**: ML model training utilities
- **Key Functions**:
  - Cross-validation implementations
  - Hyperparameter search for each model type
  - Consistent training/evaluation pipelines

#### `handle_sae_bench_saes.py`
- **Purpose**: Special handling for SAE-bench format models
- **Use Case**: Gemma-2-2b SAE compatibility

## Dataset Organization

### Dataset Structure
The repository uses numbered dataset identifiers (e.g., `100_news_fake`) where:
- Number: Unique identifier
- Name: Descriptive tag

### Dataset Categories

#### Text Classification Tasks
- News/Media: Fake news (100), clickbait (105), news categories (139-141)
- AI Detection: Human vs AI text (94, 110)
- Sentiment: Movie reviews (113), toxicity (95), spam (96)

#### Factual Knowledge Probing
- Historical Figures: Gender (5), nationality (6), profession (7)
- Geography: US states (117-119), countries (123-125), timezones (120-122)
- Entities: Art types (126-128), athletes (154-156)

#### Specialized Tasks
- GLUE Benchmarks: CoLA (87), MNLI (88), QNLI (90), etc.
- Code Classification: C (158), Python (159), HTML (160)
- Medical: Diseases (101, 145-147), cancer types (142-144)
- Binary Decisions: True/false questions, validity checks

### Data Flow

1. **Raw Text** → `generate_model_activations.py` → **Model Activations**
2. **Model Activations** → `generate_sae_activations.py` → **SAE Activations**
3. **Activations** → `train_*.py` scripts → **Probe Results**
4. **Probe Results** → `combine_results.py` → **Aggregated CSVs**
5. **Aggregated CSVs** → `plot_*.ipynb` → **Paper Figures**

## Experimental Settings

Each probe training script supports these settings:
- **normal**: Standard training conditions
- **scarcity_10/100**: Limited training data
- **class_imbalance**: 90% samples from one class
- **corrupt**: 20% label noise
- **OOD**: Train/test distribution mismatch

## Orchestration

The main orchestration script `save_sae_acts_and_train_probes.sh`:
1. Generates SAE activations for each model/layer/SAE combination
2. Trains probes in parallel (20 processes)
3. Handles memory management by restarting for each SAE

## Probing Approach Comparison

| Approach | Input Shape | Token Usage | Feature Processing | Feature Size |
|----------|-------------|-------------|-------------------|--------------|
| **Baseline Probes** | `[batch, hidden_dim]` | Last token only | Direct model activations | ~3,584 |
| **SAE Probes** | `[batch, sae_width]` | Last token only | SAE encoding + feature selection | 16k/131k/1M |
| **Multi-Token Aggregated** | `[batch, seq_len, sae_width]` | All tokens → pooling | Mean/max across sequence | 16k/131k/1M |
| **Multi-Token Concat** | `[batch, seq_len, hidden_dim]` | All tokens → concat | PCA per position + concatenate | ~5,100 |
| **Multi-Token Attention** | `[batch, seq_len, features]` | All tokens → weighted | Learned attention weights | Variable |

### Key Differences:
- **Standard Probes**: Assume final token contains all relevant information
- **Multi-Token Probes**: Test if information is distributed across the sequence
- **Memory Requirements**: Multi-token approaches require significantly more memory
- **Computational Cost**: Multi-token processing is much more expensive

## Key Design Decisions

1. **Sparse Tensors**: SAE activations saved as sparse tensors to reduce disk usage
2. **Parallelization**: Both generation and training support parallel execution
3. **Modular Design**: Clear separation between data generation, training, and analysis
4. **Reproducibility**: Fixed seeds and consistent train/test splits
5. **Multi-Token Analysis**: Comprehensive testing of sequence-level vs token-level information