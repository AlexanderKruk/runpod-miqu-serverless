import runpod
from llama_cpp import Llama
import os

llm = None

def initialize_model():
    global llm
    model_path = os.environ["MODEL_PATH"]
    n_ctx = int(os.environ.get("N_CTX", 4096))
    n_threads = int(os.environ.get("N_THREADS", 8))
    n_gpu_layers = int(os.environ.get("N_GPU_LAYERS", 83))

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        n_batch=512,
        use_mmap=True,
        use_mlock=False,
        offload_kqv=True,
        main_gpu=0,  # Explicitly specify main GPU
        tensor_split=[100],  # 100% of model on first GPU
        verbose=True  # Add verbose logging
    )

def process_input(job):
    job_input = job["input"]
    prompt = job_input.get("prompt", "")
    max_tokens = job_input.get("max_tokens", 200)
    temperature = job_input.get("temperature", 0.7)
    
    output = llm.create_completion(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False
    )
    
    return output["choices"][0]["text"]

initialize_model()
runpod.serverless.start({"handler": process_input})