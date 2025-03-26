# Use a RunPod base image with Python and CUDA 12.1 (compatible with A6000/A40)
# Check RunPod docs for potentially newer/better base images if needed.
FROM runpod/pytorch:2.1.0-cuda12.1.1-devel-ubuntu22.04
# Using -devel includes nvcc, just in case, though wheels should avoid needing it.
# You could try -runtime for a potentially smaller image if wheels work perfectly.

# Set working directory
WORKDIR /app

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies (optional, add if your handler needs more)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     some-package \
#  && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# --- Install llama-cpp-python with Pre-built CUDA Wheels ---
# This is the KEY step for faster builds and avoiding local CUDA compilation issues.
# 1. Choose a specific llama-cpp-python version (check GitHub releases for latest)
#    https://github.com/abetlen/llama-cpp-python/releases
ARG LLAMA_CPP_PYTHON_VERSION=0.2.79

# 2. Specify the CUDA version matching the base image (cu121 for CUDA 12.1)
ARG CUDA_VERSION=cu121

# 3. Install using the extra index URL for pre-built wheels
RUN pip install llama-cpp-python==${LLAMA_CPP_PYTHON_VERSION} --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/${CUDA_VERSION} --no-cache-dir

# --- Install RunPod SDK ---
RUN pip install runpod --no-cache-dir

# --- Copy Handler Code ---
COPY handler.py .

# --- Set Default Command for RunPod Serverless Worker ---
# This starts the RunPod infrastructure, which will load and run your handler.py
CMD ["python3", "-u", "/runpod_infra/runpod_serverless.py"]