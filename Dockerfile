# Use a verified RunPod base image tag with CUDA 12.1
FROM runpod/base:cuda-12.1.1-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# --- Install llama-cpp-python with Pre-built CUDA Wheels ---
ARG LLAMA_CPP_PYTHON_VERSION=0.2.79
ARG CUDA_VERSION=cu121
RUN pip install llama-cpp-python==${LLAMA_CPP_PYTHON_VERSION} --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/${CUDA_VERSION} --no-cache-dir

# --- Install RunPod SDK ---
RUN pip install runpod --no-cache-dir

# --- Copy Handler Code ---
COPY handler.py .

# --- Set Default Command (Should work with runpod/base) ---
CMD ["python3", "-u", "/runpod_infra/runpod_serverless.py"]