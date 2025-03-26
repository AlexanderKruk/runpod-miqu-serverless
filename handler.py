#!/usr/bin/env python
import runpod
from llama_cpp import Llama
import os
import time
import sys # Import sys for flushing output

# --- Constants ---
MODEL_PATH = "/runpod-volume/Midnight-Miqu-70B-v1.5.Q4_K_M.gguf"
N_CTX = 4096
N_GPU_LAYERS = -1 # Ensure this is set correctly (-1 for max possible)
N_BATCH = 512

# --- Global Model Variable ---
llm = None

# --- Utility Function for Flushing Print ---
# Sometimes in container environments, prints can be buffered.
# This helper ensures messages appear immediately in logs.
def log_message(message):
    print(message, flush=True)

# --- Initialization Function ---
def initialize():
    """Loads the model onto the GPU."""
    global llm
    log_message("-----------------------------------------")
    log_message("Starting model initialization...")
    start_time = time.time()

    log_message(f"Checking for model file at: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        # Use log_message for consistency
        log_message(f"ERROR: Model file not found at {MODEL_PATH}. "
                    "Ensure the Network Volume is correctly attached and path is accurate.")
        # Raise the error to stop initialization
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")

    log_message("Model file found.")
    log_message(f"Attempting to load Llama model with parameters:")
    log_message(f"  - model_path: {MODEL_PATH}")
    log_message(f"  - n_ctx: {N_CTX}")
    log_message(f"  - n_gpu_layers: {N_GPU_LAYERS}") # CRITICAL PARAMETER
    log_message(f"  - n_batch: {N_BATCH}")
    log_message(f"  - verbose: True") # CRITICAL FOR DEBUGGING LLAMA-CPP

    try:
        # Create the Llama instance
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS, # Pass the variable here
            n_batch=N_BATCH,
            verbose=True, # THIS IS THE MOST IMPORTANT FOR LLAMA-CPP LOGS
        )
        # If successful, log it
        log_message("Llama instance created successfully.")

        end_time = time.time()
        log_message(f"Model loading process completed in {end_time - start_time:.2f} seconds.")
        log_message("-----------------------------------------")
        log_message("Performing optional dummy inference...")
        # Optional: Perform a dummy inference to ensure everything is warm
        try:
            dummy_start_time = time.time()
            # Use the chat completion format just in case, though simple generation works too
            output = llm.create_chat_completion(
                messages = [{"role": "user", "content": "Test"}],
                max_tokens=2 # Just need a token or two
            )
            dummy_end_time = time.time()
            log_message(f"Dummy inference successful in {dummy_end_time - dummy_start_time:.2f} seconds. Output: {output['choices'][0]['message']['content']}")
        except Exception as e:
            log_message(f"Warning: Dummy inference failed: {e}")
        log_message("-----------------------------------------")

    except Exception as e:
        # Log the error clearly if Llama() fails
        log_message(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        log_message(f"ERROR: Failed to create Llama instance or load model.")
        log_message(f"Error details: {e}")
        log_message(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Propagate the error to prevent the worker from starting incorrectly
        raise e

# --- RunPod Handler Function (Keep As Is) ---
def handler(job):
    """
    Handles inference requests. Takes one job at a time (no batching here).
    """
    global llm
    if llm is None:
        log_message("ERROR in handler: Model not initialized.")
        try:
            # Attempt re-initialization, although this might indicate a bigger problem
            log_message("Attempting to re-initialize model within handler...")
            initialize()
        except Exception as e:
            log_message(f"ERROR: Re-initialization failed: {e}")
            return {"error": f"Model initialization failed during request: {e}"}
        if llm is None: # Still not loaded after attempt
             log_message("ERROR: Model failed to load and is unavailable after re-attempt.")
             return {"error": "Model failed to load and is unavailable."}

    job_input = job.get('input', {})
    prompt = job_input.get('prompt')

    if not prompt:
        log_message("ERROR in handler: No 'prompt' provided in input.")
        return {"error": "No 'prompt' provided in input."}

    # --- Generation Parameters ---
    max_tokens = job_input.get('max_tokens', 512)
    temperature = job_input.get('temperature', 0.7)
    top_p = job_input.get('top_p', 0.95)
    stop = job_input.get('stop', ["</s>", "<|im_end|>"])

    log_message(f"Received job {job.get('id', 'N/A')}. Starting inference...")
    log_message(f"Params: max_tokens={max_tokens}, temp={temperature}, top_p={top_p}, stop={stop}")
    handler_start_time = time.time()

    try:
        # Using create_chat_completion is often preferred for instruction-tuned models
        output = llm.create_chat_completion(
            messages = [
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            # stream=False # Default is False for runsync
        )
        result_text = output['choices'][0]['message']['content'].strip()

        handler_end_time = time.time()
        log_message(f"Job {job.get('id', 'N/A')} completed in {handler_end_time - handler_start_time:.2f} seconds.")

        # Return the generated text
        return {"text": result_text}

    except Exception as e:
        log_message(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        log_message(f"ERROR during inference for job {job.get('id', 'N/A')}: {e}")
        log_message(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return {"error": f"Inference failed: {e}"}

# --- RunPod Serverless Entrypoint ---
# Initialize the model when the worker starts, before accepting jobs.
try:
    initialize() # Run the updated initialization function
    log_message("Initialization complete. Starting RunPod serverless worker...")
    # Start the RunPod serverless worker, passing the handler function
    runpod.serverless.start({"handler": handler})
except Exception as e:
    # If initialization fails catastrophically, log it clearly.
    log_message(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    log_message(f"FATAL: Initialization failed during startup: {e}")
    log_message(f"Worker failed to start.")
    log_message(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # Optionally exit to signal failure more strongly, but RunPod might restart it anyway
    # sys.exit(1)