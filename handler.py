#!/usr/bin/env python
import runpod
from llama_cpp import llama_info # Import llama_info specifically
from llama_cpp import Llama
import os
import time
import sys

# --- Constants ---
MODEL_PATH = "/runpod-volume/Midnight-Miqu-70B-v1.5.Q4_K_M.gguf"
N_CTX = 4096
N_GPU_LAYERS = -1 # Target: Use GPU
N_BATCH = 512

# --- Global Model Variable ---
llm = None

# --- Utility Function for Flushing Print ---
def log_message(message):
    print(message, flush=True)

# --- Initialization Function ---
def initialize():
    """Loads the model onto the GPU."""
    global llm
    log_message("-----------------------------------------")
    log_message("Starting model initialization...")
    start_time = time.time()

    # --- >> NEW: Check llama-cpp installation before loading model << ---
    log_message("Checking llama_cpp library info...")
    try:
        info = llama_info()
        log_message(f"llama_cpp info: {info}")
        # Explicitly check for cuBLAS/CUDA support reported by the library
        if info.get('ggml_build_cublas', False) or info.get('ggml_build_cuda', False):
             log_message("SUCCESS: llama_cpp reports CUDA/cuBLAS support IS enabled.")
        else:
             log_message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             log_message("ERROR: llama_cpp reports CUDA/cuBLAS support IS NOT enabled in the installed package.")
             log_message("The build/installation likely failed to include GPU support.")
             log_message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             # You could optionally raise an error here to stop if GPU is mandatory
             # raise RuntimeError("llama-cpp-python installed without GPU support!")
    except Exception as e:
        log_message(f"ERROR: Could not retrieve llama_cpp info: {e}")
        # Depending on the error, you might want to raise it
    log_message("-----------------------------------------")
    # --- >> End of New Check << ---


    log_message(f"Checking for model file at: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        log_message(f"ERROR: Model file not found at {MODEL_PATH}.")
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")

    log_message("Model file found.")
    log_message(f"Attempting to load Llama model with parameters:")
    log_message(f"  - model_path: {MODEL_PATH}")
    log_message(f"  - n_ctx: {N_CTX}")
    log_message(f"  - n_gpu_layers: {N_GPU_LAYERS}")
    log_message(f"  - n_batch: {N_BATCH}")
    log_message(f"  - verbose: True")

    try:
        # --- >> Optional: Try loading with n_gpu_layers=0 first? << ---
        # If the check above passed but it still hangs/crashes here, you could
        # uncomment the block below to see if it loads on CPU ONLY. If it does,
        # the issue is purely GPU initialization. If it still fails, the model
        # file or basic loading might be the problem.
        # log_message("DEBUG: Attempting preliminary load with n_gpu_layers=0...")
        # try:
        #     temp_llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, n_gpu_layers=0, verbose=True)
        #     log_message("DEBUG: Preliminary load with n_gpu_layers=0 SUCCEEDED.")
        #     del temp_llm # clean up memory
        # except Exception as e_cpu:
        #     log_message(f"DEBUG: Preliminary load with n_gpu_layers=0 FAILED: {e_cpu}")
        #     raise e_cpu # Raise if CPU load fails, likely bigger problem
        # log_message("-----------------------------------------")
        # --- >> End Optional CPU Load Check << ---

        log_message("Proceeding with GPU offload attempt (n_gpu_layers={N_GPU_LAYERS})...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            n_batch=N_BATCH,
            verbose=True,
        )
        log_message("Llama instance created successfully (GPU load expected).")

        end_time = time.time()
        log_message(f"Model loading process completed in {end_time - start_time:.2f} seconds.")
        log_message("-----------------------------------------")
        # ... (rest of the function remains the same - dummy inference etc.)

    except Exception as e:
        log_message(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        log_message(f"ERROR: Failed during Llama instance creation or model loading.")
        log_message(f"Error details: {e}")
        log_message(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise e

# --- Handler function and RunPod start remain the same ---
# ...
def handler(job):
# ...
# ... (rest of the file) ...

try:
    initialize()
    log_message("Initialization complete. Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
except Exception as e:
    log_message(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    log_message(f"FATAL: Initialization failed during startup: {e}")
    log_message(f"Worker failed to start.")
    log_message(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")