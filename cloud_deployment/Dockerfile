FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Set up working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY . .

# Install dependencies using uv
RUN uv venv
RUN uv pip sync uv.lock

# Set up HuggingFace cache
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create data directories
RUN mkdir -p data/sae_activations_gemma-2-9b/normal_setting

# Entry point using uv
ENTRYPOINT ["uv", "run"]
CMD ["python", "generate_sae_activations.py", "--model_name", "gemma-2-9b", "--setting", "normal", "--device", "cuda:0"]