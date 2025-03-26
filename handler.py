#!/usr/bin/env python
import runpod
from llama_cpp import Llama
import os
import time

# --- Constants ---
MODEL_PATH = "/runpod-volume/Midnight-Miqu-70B-v1.5.Q4_K_M.gguf"
# Adjust n_ctx based on your needs, 4096 is common for Miqu.
# Check model card if possible for recommended context size.
N_CTX = 4096
# Offload all possible layers to GPU. -1 means all layers.
N_GPU_LAYERS = -1
# For Serverless, internal batching within llama.cpp can still be useful.
# This is NOT request batching, but how llama.cpp processes tokens internally.
# 512 is a common default. Adjust if needed based on performance/memory.
N_BATCH = 512

# --- Global Model Variable ---
llm = None

# --- Initialization Function ---
def initialize():
    """Loads the model onto the GPU."""
    global llm
    print("Initializing model...")
    start_time = time.time()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. "
                                "Ensure the Network Volume is correctly attached "
                                "and the path is accurate.")

    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            n_batch=N_BATCH,
            verbose=True,  # Set to False for less detailed logs in production
            # Add other llama.cpp parameters if needed, e.g., seed=-1 for random
        )
        end_time = time.time()
        print(f"Model loaded successfully in {end_time - start_time:.2f} seconds.")
        # Optional: Perform a dummy inference to ensure everything is warm
        try:
            llm("Test prompt.", max_tokens=1)
            print("Dummy inference successful.")
        except Exception as e:
            print(f"Warning: Dummy inference failed: {e}")

    except Exception as e:
        print(f"Error loading model: {e}")
        # Propagate the error to prevent the worker from starting incorrectly
        raise e

# --- RunPod Handler Function ---
def handler(job):
    """
    Handles inference requests. Takes one job at a time (no batching here).
    """
    global llm
    if llm is None:
        # This should ideally not happen if initialization worked,
        # but good as a safeguard.
        print("Model not initialized. Attempting to initialize...")
        try:
            initialize()
        except Exception as e:
            return {"error": f"Model initialization failed during request: {e}"}
        if llm is None: # Still not loaded after attempt
             return {"error": "Model failed to load and is unavailable."}


    job_input = job.get('input', {})
    prompt = job_input.get('prompt')

    if not prompt:
        return {"error": "No 'prompt' provided in input."}

    # --- Generation Parameters ---
    # Set defaults and allow overrides from job input
    max_tokens = job_input.get('max_tokens', 512)
    temperature = job_input.get('temperature', 0.7)
    top_p = job_input.get('top_p', 0.95)
    stop = job_input.get('stop', ["</s>", "<|im_end|>"]) # Common stop tokens
    # Add other parameters as needed (e.g., top_k, frequency_penalty, etc.)

    print(f"Received job {job['id']}. Starting inference...")
    start_time = time.time()

    try:
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            # Add other generation parameters here if needed
            # echo=False # Don't echo the prompt in the output
        )
        result_text = output['choices'][0]['text'].strip()

        end_time = time.time()
        print(f"Job {job['id']} completed in {end_time - start_time:.2f} seconds.")

        # Return the generated text
        return {"text": result_text}

    except Exception as e:
        print(f"Error during inference for job {job['id']}: {e}")
        return {"error": f"Inference failed: {e}"}

# --- RunPod Serverless Entrypoint ---
# Initialize the model when the worker starts, before accepting jobs.
try:
    initialize()
    # Start the RunPod serverless worker, passing the handler function
    runpod.serverless.start({"handler": handler})
except Exception as e:
    # If initialization fails catastrophically, log it.
    # RunPod's infrastructure might restart the worker.
    print(f"FATAL: Initialization failed: {e}")
    # Optionally, exit or raise to signal failure clearly
    # raise SystemExit(1) # This might cause rapid restarts, use with caution