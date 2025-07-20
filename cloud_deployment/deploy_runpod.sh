#!/bin/bash
# Deployment script for RunPod or similar GPU cloud providers
# Uses uv for dependency management

set -e

# Configuration
MODEL_NAME="${MODEL_NAME:-gemma-2-9b}"
SETTING="${SETTING:-normal}"
DEVICE="${DEVICE:-cuda:0}"
STORAGE_BACKEND="${STORAGE_BACKEND:-s3}"
STORAGE_BUCKET="${STORAGE_BUCKET:-sae-probes-activations}"

echo "SAE Activation Generation Deployment"
echo "===================================="
echo "Model: $MODEL_NAME"
echo "Setting: $SETTING"
echo "Device: $DEVICE"
echo "Storage: $STORAGE_BACKEND ($STORAGE_BUCKET)"
echo ""

# Step 1: Install system dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y git curl python3-pip

# Step 2: Install uv
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# Step 3: Clone repository (or use mounted volume)
if [ ! -d "/workspace/SAE-Probes" ]; then
    echo "Cloning repository..."
    cd /workspace
    git clone https://github.com/JoshEngels/SAE-Probes.git
fi

cd /workspace/SAE-Probes

# Step 4: Set up Python environment with uv
echo "Setting up Python environment with uv..."
uv venv
source .venv/bin/activate

# Step 5: Install dependencies
echo "Installing dependencies..."
uv pip sync uv.lock

# Step 6: Install cloud storage dependencies if needed
if [ "$STORAGE_BACKEND" = "s3" ]; then
    echo "Installing AWS S3 support..."
    uv pip install boto3
elif [ "$STORAGE_BACKEND" = "gcs" ]; then
    echo "Installing Google Cloud Storage support..."
    uv pip install google-cloud-storage
fi

# Step 7: Create necessary directories
echo "Creating data directories..."
mkdir -p data/sae_activations_${MODEL_NAME}/normal_setting
mkdir -p data/sae_activations_${MODEL_NAME}/OOD_setting
mkdir -p data/sae_activations_${MODEL_NAME}/scarcity_setting
mkdir -p data/sae_activations_${MODEL_NAME}/class_imbalance_setting

# Step 8: Download model activations if not present
if [ ! -d "data/model_activations_${MODEL_NAME}" ]; then
    echo "Downloading model activations..."
    # Add download commands here if you have direct links
    # wget [URL] -O data/model_activations_${MODEL_NAME}.tar.gz
    # tar -xzf data/model_activations_${MODEL_NAME}.tar.gz -C data/
fi

# Step 9: Run the generation with cloud support
echo "Starting SAE activation generation..."
cd cloud_deployment
uv run python run_cloud_generation.py \
    --model_name $MODEL_NAME \
    --setting $SETTING \
    --device $DEVICE \
    --storage $STORAGE_BACKEND \
    --bucket $STORAGE_BUCKET \
    --max_runs 100

echo "Deployment complete!"