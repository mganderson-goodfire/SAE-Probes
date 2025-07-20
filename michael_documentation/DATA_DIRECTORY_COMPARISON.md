# Data Directory Comparison: ./data vs ./raw_text_datasets

## Overview

The repository has two main data directories with distinct roles in the SAE probing pipeline:
- `./data/` - Contains processed results and master catalog
- `./raw_text_datasets/` - Contains source datasets, baseline results, and archived activations

## Directory Structure Breakdown

### ./data/ Directory (Processed Results)

```
data/
├── OOD data/                           # 8 OOD test files
├── consolidated_probing_gemma-2-9b/    # Processed SAE probing results
└── probing_datasets_MASTER.csv        # Master catalog (155+ datasets)
```

**Purpose**: Stores processed experimental results and dataset catalog

**Key Contents**:
- **consolidated_probing_gemma-2-9b/**: 461 pickle files containing:
  - Attention probing results (`*_attn_probing.pkl`)
  - Baseline results (`*_baseline_*.pkl`)
  - SAE activation results with different configurations
  - Both regular and binarized versions
- **probing_datasets_MASTER.csv**: Central catalog mapping dataset names, file locations, and metadata

### ./raw_text_datasets/ Directory (Source Data & Archives)

```
raw_text_datasets/
├── cleaned_data/                       # 155 source CSV datasets
├── OOD data/                          # 8 OOD files + HuggingFace caches
├── baseline_results_gemma-2-9b/       # Baseline algorithm results
├── dataset_investigate/               # Dataset investigation files
├── model_activations_gemma-2-9b.tar.gz      # Compressed model activations
├── sae_activations_gemma-2-9b_OOD.tar.gz    # Compressed SAE activations
└── [other compressed archives]
```

**Purpose**: Source data repository with raw datasets, baseline results, and archived activations

## Data Flow Relationship

```
Source Data → Processing → Results Storage
```

1. **Source Datasets**: `raw_text_datasets/cleaned_data/` (155 CSV files)
   ↓
2. **Model Processing**: Extract activations → Store in compressed archives
   ↓
3. **Probing Experiments**: Run baseline and SAE probes
   ↓
4. **Results Storage**: 
   - Baseline results → `raw_text_datasets/baseline_results_gemma-2-9b/`
   - SAE results → `data/consolidated_probing_gemma-2-9b/`

## Key Differences

### File Formats
- **raw_text_datasets/**: Primarily CSV files (source data) and compressed archives
- **data/**: Primarily pickle files (processed results)

### Organization
- **raw_text_datasets/**: Organized by data type and processing stage
- **data/**: Organized by experiment type and model

### Content Overlap
- **OOD Data**: Both directories contain the same 8 OOD CSV files
- **Additional in raw_text_datasets/**: HuggingFace dataset caches and translation files

## Dataset Categories (from MASTER.csv)

### Binary Classification Tasks (113 datasets)
- Text classification: News, sentiment, spam detection
- Factual knowledge: Historical figures, geography, entities
- Specialized: GLUE benchmarks, code classification, medical data

### Examples by Category:
- **News/Media**: `100_news_fake.csv`, `105_click_bait.csv`
- **Knowledge Probing**: `5_hist_fig_ismale.csv`, `117_us_state_FL.csv`
- **NLP Benchmarks**: `87_glue_cola.csv`, `90_glue_qnli.csv`
- **Code**: `158_code_C.csv`, `159_code_Python.csv`

## Usage Patterns

### During Development/Experimentation
```bash
# Source data access
utils_data.py → raw_text_datasets/cleaned_data/

# Master catalog reference
probing_datasets_MASTER.csv → data/probing_datasets_MASTER.csv

# Results storage
training scripts → data/consolidated_probing_gemma-2-9b/
```

### For Reproduction
```bash
# Extract pre-computed activations
tar -xzf raw_text_datasets/model_activations_*.tar.gz

# Access baseline results
raw_text_datasets/baseline_results_gemma-2-9b/

# Access processed SAE results
data/consolidated_probing_gemma-2-9b/
```

## Storage Considerations

### Disk Usage
- **raw_text_datasets/**: ~Several GB (compressed archives)
- **data/**: ~Hundreds of MB (pickle files)

### File Counts
- **raw_text_datasets/cleaned_data/**: 155 CSV files
- **data/consolidated_probing_gemma-2-9b/**: 461 pickle files

## Practical Implications

### For M3 MacBook Pro Users
- **Source datasets** (`raw_text_datasets/cleaned_data/`) can be used directly
- **Baseline results** available in `raw_text_datasets/baseline_results_gemma-2-9b/`
- **Pre-computed SAE results** in `data/consolidated_probing_gemma-2-9b/`

### For GPU Cloud Users
- Generate fresh activations → Store in `raw_text_datasets/`
- Run experiments → Results go to `data/`
- Archive results → Compress back to `raw_text_datasets/`

## Configuration Notes

### Path References in Code
- `utils_data.py` expects data in `../SAE-Probing/data/` (relative path)
- This suggests the original repository structure expected these directories

### Makefile Targets
- `make setup` extracts archives from `raw_text_datasets/`
- Training targets work with both directories as needed