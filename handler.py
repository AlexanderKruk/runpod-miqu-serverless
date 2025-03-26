import runpod
from llama_cpp import Llama
import os

llm = None

def initialize_model():
    global llm
    model_config = {
        "model_path": os.environ["MODEL_PATH"],
        "n_ctx": int(os.environ["N_CTX"]),
        "n_gpu_layers": int(os.environ["N_GPU_LAYERS"]),
        "n_batch": int(os.environ["N_BATCH"]),
        "main_gpu": int(os.environ["MAIN_GPU"]),
        "tensor_split": [float(os.environ["GPU_SPLIT"])],
        "n_threads": 8,
        "verbose": True,
        "offload_kqv": True,
        "mul_mat_q": True,
        "use_mlock": False,
        "use_mmap": True
    }
    
    print("Initializing model with config:", model_config)
    llm = Llama(**model_config)
    print("VRAM Usage:", llm._ctx.mem_usage())  # Add memory usage report

def process_input(job):
    job_input = job["input"]
    try:
        result = llm.create_completion(
            prompt=job_input["prompt"],
            max_tokens=job_input.get("max_tokens", 200),
            temperature=job_input.get("temperature", 0.7),
            stream=False
        )
        return result["choices"][0]["text"]
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    initialize_model()
    runpod.serverless.start({"handler": process_input})