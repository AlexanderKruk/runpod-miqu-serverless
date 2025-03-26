# Use the official NVIDIA CUDA image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Install Python3, Pip, Git, AND essential build tools AND Upgrade Pip
# Added build-essential, cmake as they are needed for building from source if wheels fail
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    build-essential \
    cmake \
 && python3 -m pip install --upgrade pip \
 && rm -rf /var/lib/apt/lists/*

# --- Install llama-cpp-python with CUDA ---

# Set Environment Variables for building from source (as a fallback)
# These tell the llama-cpp-python setup process to build with CUDA support
# using the tools available in the nvidia/cuda base image.
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
ENV FORCE_CMAKE=1

# Arguments for version and CUDA compatibility (match base image)
ARG LLAMA_CPP_PYTHON_VERSION=0.2.79
ARG CUDA_VERSION=cu121

# Install:
# 1. Uninstall any previous attempts (clean slate)
# 2. Install wheel package (helper for pip)
# 3. Attempt install using --extra-index-url for pre-built wheels FIRST.
#    If this finds a matching wheel, it should be fast and have CUDA.
# 4. Use --force-reinstall and --no-cache-dir to avoid cached bad installs.
# 5. If wheel download fails, pip *should* fall back to building from source,
#    using the CMAKE_ARGS we set above.
RUN python3 -m pip uninstall llama-cpp-python -y || echo "llama-cpp-python not previously installed"
RUN python3 -m pip install wheel
RUN python3 -m pip install llama-cpp-python==${LLAMA_CPP_PYTHON_VERSION} \
    --force-reinstall --no-cache-dir \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/${CUDA_VERSION}

# --- Add a Build-Time Check ---
# This runs during docker build. If it fails here, the install was wrong.
RUN echo "Verifying llama-cpp-python install during build..." && \
    python3 -c "import llama_cpp; info = llama_cpp.llama_info(); print(info); assert info.get('ggml_build_cublas', False) or info.get('ggml_build_cuda', False), 'ERROR: Build-time check FAILED - llama-cpp-python installed without CUDA/cuBLAS support!'" || \
    (echo "Build-time check failed, exiting." && exit 1)
# --- End Build-Time Check ---

# --- Install RunPod SDK ---
RUN python3 -m pip install runpod --no-cache-dir

# --- Copy Handler Code ---
COPY handler.py .

# --- Set Default Command: Execute handler.py directly ---
CMD ["python3", "-u", "handler.py"]