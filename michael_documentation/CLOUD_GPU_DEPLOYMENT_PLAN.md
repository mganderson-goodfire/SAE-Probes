# Cloud GPU Deployment Plan for SAE Activation Generation

## Overview
This document outlines approaches for deploying `generate_sae_activations.py` to cloud GPU providers to generate the missing normal setting SAE activations.

## Key Requirements
- **GPU Memory**: 24GB minimum (RTX 4090, A10, A40, A100)
- **Dependencies**: PyTorch, sae_lens, transformers, HuggingFace models
- **Storage**: ~50-100GB for generated activations
- **Runtime**: ~100 runs of the script (due to CUDA memory leakage)

## Approach 1: Container-Based Deployment (Recommended)

### Docker Container Setup
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install dependencies
RUN pip install --upgrade pip
RUN pip install transformer_lens sae_lens transformers datasets torch xgboost sae_bench scikit-learn natsort boto3

# Copy code
WORKDIR /app
COPY . /app/

# Set up HuggingFace cache
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

# Entry point
CMD ["python", "generate_sae_activations.py"]
```

### Cloud Storage Integration

#### AWS S3 Setup
```python
# Add to generate_sae_activations.py
import boto3
import os

def upload_to_s3(local_path, bucket_name, s3_path):
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket_name, s3_path)

def sync_activations_to_s3(model_name, setting):
    """Sync generated activations to S3 after each batch"""
    local_dir = f"data/sae_activations_{model_name}/{setting}_setting/"
    s3_bucket = os.environ.get('S3_BUCKET', 'sae-probes-activations')
    
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            if file.endswith('.pt'):
                local_path = os.path.join(root, file)
                s3_path = local_path.replace('data/', '')
                upload_to_s3(local_path, s3_bucket, s3_path)
```

#### Google Cloud Storage
```python
from google.cloud import storage

def upload_to_gcs(local_path, bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

def sync_activations_to_gcs(model_name, setting):
    """Sync generated activations to GCS after each batch"""
    local_dir = f"data/sae_activations_{model_name}/{setting}_setting/"
    bucket_name = os.environ.get('GCS_BUCKET', 'sae-probes-activations')
    
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            if file.endswith('.pt'):
                local_path = os.path.join(root, file)
                blob_name = local_path.replace('data/', '')
                upload_to_gcs(local_path, bucket_name, blob_name)
```

## Approach 2: Script-Based Deployment with rsync

### Setup Script (setup_gpu_instance.sh)
```bash
#!/bin/bash
# Run on fresh GPU instance

# Install Python and dependencies
sudo apt-get update
sudo apt-get install -y python3-pip git rsync

# Clone repository
git clone https://github.com/JoshEngels/SAE-Probes.git
cd SAE-Probes

# Install Python dependencies
pip install transformer_lens sae_lens transformers datasets torch xgboost sae_bench scikit-learn natsort

# Create data directories
mkdir -p data/sae_activations_gemma-2-9b/normal_setting

# Download model activations (if needed)
# wget [dropbox_link] -O data/model_activations_gemma-2-9b.tar.gz
# tar -xzf data/model_activations_gemma-2-9b.tar.gz -C data/
```

### Orchestration Script (run_sae_generation.sh)
```bash
#!/bin/bash
# Modified version of save_sae_acts_and_train_probes.sh for cloud deployment

MODEL_NAME="gemma-2-9b"
SETTING="normal"
DEVICE="cuda:0"
S3_BUCKET="s3://your-bucket/sae-activations"
LOCAL_DATA_DIR="data/sae_activations_${MODEL_NAME}/${SETTING}_setting"

# Function to sync to S3 after each run
sync_to_s3() {
    aws s3 sync $LOCAL_DATA_DIR $S3_BUCKET/$MODEL_NAME/$SETTING_setting/ --exclude "*.tmp"
}

# Run generation with periodic syncing
for i in {1..100}
do
    echo "Run $i of 100..."
    python3 generate_sae_activations.py \
        --model_name $MODEL_NAME \
        --setting $SETTING \
        --device $DEVICE
    
    # Sync to S3 every 5 runs
    if [ $((i % 5)) -eq 0 ]; then
        echo "Syncing to S3..."
        sync_to_s3
    fi
    
    # Optional: Clear GPU cache
    nvidia-smi --gpu-reset
done

# Final sync
sync_to_s3
```

## Approach 3: Kubernetes Job (For Scale)

### Kubernetes Job Definition
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: sae-generation-gemma-normal
spec:
  parallelism: 4  # Run 4 pods in parallel
  completions: 100  # Total runs needed
  template:
    spec:
      containers:
      - name: sae-generator
        image: your-registry/sae-probes:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
        env:
        - name: MODEL_NAME
          value: "gemma-2-9b"
        - name: SETTING
          value: "normal"
        - name: S3_BUCKET
          value: "sae-probes-activations"
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: access-key
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: secret-key
        command: ["python", "generate_sae_activations.py"]
        args: ["--model_name", "$(MODEL_NAME)", "--setting", "$(SETTING)", "--device", "cuda:0"]
      restartPolicy: OnFailure
      nodeSelector:
        gpu-type: "rtx4090"
```

## Approach 4: Spot Instance Automation

### AWS Spot Fleet Configuration
```python
# spot_fleet_runner.py
import boto3
import time

def create_spot_fleet(num_instances=4):
    ec2 = boto3.client('ec2')
    
    user_data = '''#!/bin/bash
    # Clone repo and setup
    git clone https://github.com/your-fork/SAE-Probes.git
    cd SAE-Probes
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Run generation with instance ID for uniqueness
    INSTANCE_ID=$(ec2-metadata --instance-id | cut -d " " -f 2)
    python generate_sae_activations.py \
        --model_name gemma-2-9b \
        --setting normal \
        --device cuda:0 \
        --randomize_order
    
    # Sync results to S3
    aws s3 sync data/sae_activations_gemma-2-9b/normal_setting/ \
        s3://your-bucket/sae-activations/gemma-2-9b/normal_setting/
    
    # Terminate instance when done
    shutdown -h now
    '''
    
    response = ec2.request_spot_fleet(
        SpotFleetRequestConfig={
            'IamFleetRole': 'arn:aws:iam::account:role/fleet-role',
            'TargetCapacity': num_instances,
            'SpotPrice': '0.50',
            'LaunchSpecifications': [{
                'ImageId': 'ami-0123456789',  # Deep Learning AMI
                'InstanceType': 'g5.xlarge',
                'KeyName': 'your-key',
                'UserData': user_data,
                'BlockDeviceMappings': [{
                    'DeviceName': '/dev/sda1',
                    'Ebs': {'VolumeSize': 100}
                }]
            }]
        }
    )
    return response
```

## Storage Optimization

### Compression Before Upload
```python
def compress_and_upload(local_path, remote_path):
    """Compress activations before uploading to save bandwidth"""
    import gzip
    import shutil
    
    # Compress the file
    compressed_path = local_path + '.gz'
    with open(local_path, 'rb') as f_in:
        with gzip.open(compressed_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Upload compressed file
    upload_to_s3(compressed_path, bucket_name, remote_path + '.gz')
    
    # Clean up
    os.remove(compressed_path)
```

### Batch Processing Script
```python
# batch_processor.py
import os
import sys
from generate_sae_activations import process_model_setting

def main():
    # Get task ID from environment or command line
    task_id = int(os.environ.get('TASK_ID', sys.argv[1]))
    total_tasks = int(os.environ.get('TOTAL_TASKS', '100'))
    
    # Deterministic assignment of work
    import random
    random.seed(42)
    
    # Generate all dataset-SAE combinations
    all_tasks = []
    for dataset in datasets:
        for sae_id in sae_ids:
            all_tasks.append((dataset, sae_id))
    
    # Shuffle deterministically
    random.shuffle(all_tasks)
    
    # Get this worker's subset
    tasks_per_worker = len(all_tasks) // total_tasks
    start_idx = task_id * tasks_per_worker
    end_idx = start_idx + tasks_per_worker
    
    my_tasks = all_tasks[start_idx:end_idx]
    
    # Process assigned tasks
    for dataset, sae_id in my_tasks:
        process_single_combination(dataset, sae_id)

if __name__ == "__main__":
    main()
```

## Monitoring and Recovery

### Progress Tracking
```python
def track_progress(model_name, setting):
    """Track which activations have been generated"""
    import json
    
    progress_file = f"progress_{model_name}_{setting}.json"
    completed = set()
    
    # Check existing files
    activation_dir = f"data/sae_activations_{model_name}/{setting}_setting/"
    if os.path.exists(activation_dir):
        for file in os.listdir(activation_dir):
            if file.endswith('_X_train_sae.pt'):
                dataset_sae = file.replace('_X_train_sae.pt', '')
                completed.add(dataset_sae)
    
    # Save progress
    with open(progress_file, 'w') as f:
        json.dump(list(completed), f)
    
    return completed
```

## Recommended Deployment Strategy

### For Quick Testing (1-2 SAEs):
1. Use Google Colab Pro+ with manual monitoring
2. Save results to Google Drive

### For Full Normal Setting Generation:
1. **Provider**: RunPod or Vast.ai
2. **Setup**: 4x RTX 4090 instances
3. **Storage**: AWS S3 or Google Cloud Storage
4. **Approach**: Container-based with automatic syncing
5. **Estimated Time**: 2-3 days
6. **Estimated Cost**: $100-150

### Steps:
1. Fork repository and add cloud storage integration
2. Build Docker container with dependencies
3. Deploy to chosen GPU provider
4. Run orchestration script with periodic syncing
5. Download final results to local machine

### Alternative: Partial Generation
Generate only the SAEs needed for key experiments:
- Layer 20 with width_16k, width_131k, width_1m
- Focus on 20-30 most important datasets
- Reduces time to ~8-12 hours