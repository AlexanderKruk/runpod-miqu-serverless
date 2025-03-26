# Use the official NVIDIA CUDA image as RunPod images seem unavailable
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Install Python3, Pip, Git AND Upgrade Pip in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
&& python3 -m pip install --upgrade pip \
&& rm -rf /var/lib/apt/lists/*

# --- Install llama-cpp-python with Pre-built CUDA Wheels ---
ARG LLAMA_CPP_PYTHON_VERSION=0.2.79
ARG CUDA_VERSION=cu121
# Ensure wheel package is installed
RUN python3 -m pip install wheel
RUN python3 -m pip install llama-cpp-python==${LLAMA_CPP_PYTHON_VERSION} --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/${CUDA_VERSION} --no-cache-dir

# --- Install RunPod SDK ---
RUN python3 -m pip install runpod --no-cache-dir

# --- Copy Handler Code ---
COPY handler.py .

# --- Set Default Command: Execute handler.py directly ---
# The handler.py script itself calls runpod.serverless.start() using the installed SDK
CMD ["python3", "-u", "handler.py"]