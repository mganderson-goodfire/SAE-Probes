#!/usr/bin/env python3
"""
Cloud deployment wrapper for SAE activation generation with uv.
Adds cloud storage support and progress tracking.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import from main project
sys.path.append(str(Path(__file__).parent.parent))

# Cloud storage imports (optional based on environment)
try:
    import boto3
    HAS_S3 = True
except ImportError:
    HAS_S3 = False

try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False


class CloudSAEGenerator:
    def __init__(self, storage_backend='s3', bucket_name=None):
        self.storage_backend = storage_backend
        self.bucket_name = bucket_name or os.environ.get('STORAGE_BUCKET', 'sae-probes-activations')
        self.progress_file = 'generation_progress.json'
        self.completed_tasks = self.load_progress()
        
    def load_progress(self):
        """Load previously completed tasks"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def save_progress(self):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(list(self.completed_tasks), f, indent=2)
    
    def upload_to_s3(self, local_path, s3_key):
        """Upload file to S3"""
        if not HAS_S3:
            print(f"Warning: boto3 not installed, skipping S3 upload of {local_path}")
            return
            
        s3 = boto3.client('s3')
        try:
            s3.upload_file(str(local_path), self.bucket_name, s3_key)
            print(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
        except Exception as e:
            print(f"Error uploading to S3: {e}")
    
    def upload_to_gcs(self, local_path, blob_name):
        """Upload file to Google Cloud Storage"""
        if not HAS_GCS:
            print(f"Warning: google-cloud-storage not installed, skipping GCS upload of {local_path}")
            return
            
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(blob_name)
        try:
            blob.upload_from_filename(str(local_path))
            print(f"Uploaded {local_path} to gs://{self.bucket_name}/{blob_name}")
        except Exception as e:
            print(f"Error uploading to GCS: {e}")
    
    def sync_activations(self, model_name, setting):
        """Sync generated activations to cloud storage"""
        local_dir = Path(f"data/sae_activations_{model_name}/{setting}_setting")
        
        if not local_dir.exists():
            print(f"Warning: {local_dir} does not exist, nothing to sync")
            return
        
        # Find all .pt files
        for pt_file in local_dir.glob("*.pt"):
            relative_path = pt_file.relative_to("data")
            cloud_path = f"sae-probes/{relative_path}"
            
            if self.storage_backend == 's3':
                self.upload_to_s3(pt_file, cloud_path)
            elif self.storage_backend == 'gcs':
                self.upload_to_gcs(pt_file, cloud_path)
            else:
                print(f"Unknown storage backend: {self.storage_backend}")
    
    def run_generation(self, model_name, setting, device='cuda:0', max_runs=100):
        """Run the generation process with progress tracking and syncing"""
        
        for run_num in range(1, max_runs + 1):
            task_id = f"{model_name}_{setting}_run_{run_num}"
            
            if task_id in self.completed_tasks:
                print(f"Skipping completed task: {task_id}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Run {run_num}/{max_runs} - {model_name} {setting} setting")
            print(f"Timestamp: {datetime.now().isoformat()}")
            print(f"{'='*60}\n")
            
            # Run the generation script using uv
            cmd = [
                "uv", "run", "python", "generate_sae_activations.py",
                "--model_name", model_name,
                "--setting", setting,
                "--device", device
            ]
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(result.stdout)
                
                # Mark as completed
                self.completed_tasks.add(task_id)
                self.save_progress()
                
                # Sync to cloud storage every 5 runs
                if run_num % 5 == 0:
                    print(f"\nSyncing to {self.storage_backend}...")
                    self.sync_activations(model_name, setting)
                    
            except subprocess.CalledProcessError as e:
                print(f"Error in run {run_num}: {e}")
                print(f"Stdout: {e.stdout}")
                print(f"Stderr: {e.stderr}")
                
                # Check if it's the expected completion message
                if "All SAE activations" in str(e.stdout):
                    print("All activations have been generated!")
                    break
                else:
                    # Wait a bit before retrying
                    time.sleep(10)
        
        # Final sync
        print(f"\nFinal sync to {self.storage_backend}...")
        self.sync_activations(model_name, setting)
        print("\nGeneration complete!")


def main():
    parser = argparse.ArgumentParser(description="Cloud deployment wrapper for SAE generation")
    parser.add_argument("--model_name", type=str, default="gemma-2-9b",
                        choices=["gemma-2-9b", "llama-3.1-8b", "gemma-2-2b"])
    parser.add_argument("--setting", type=str, default="normal",
                        choices=["normal", "scarcity", "imbalance", "OOD"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--storage", type=str, default="s3",
                        choices=["s3", "gcs", "none"])
    parser.add_argument("--bucket", type=str, default=None,
                        help="Storage bucket name (uses STORAGE_BUCKET env var if not set)")
    parser.add_argument("--max_runs", type=int, default=100,
                        help="Maximum number of runs (default: 100)")
    
    args = parser.parse_args()
    
    # Create generator instance
    generator = CloudSAEGenerator(
        storage_backend=args.storage if args.storage != 'none' else None,
        bucket_name=args.bucket
    )
    
    # Run generation
    generator.run_generation(
        model_name=args.model_name,
        setting=args.setting,
        device=args.device,
        max_runs=args.max_runs
    )


if __name__ == "__main__":
    main()