import runpod
from llama_cpp import Llama
import os

llm = None

def initialize_model():
    global llm
    try:
        model_config = {
            "model_path": os.environ["MODEL_PATH"],
            "n_ctx": int(os.environ["N_CTX"]),
            "n_gpu_layers": int(os.environ["N_GPU_LAYERS"]),
            "n_batch": int(os.environ["N_BATCH"]),
            "main_gpu": int(os.environ["MAIN_GPU"]),
            "tensor_split": [float(os.environ["GPU_SPLIT"])],
            "n_threads": 8,
            "verbose": False,  # Disable verbose logging for production
            "offload_kqv": True,
            "mul_mat_q": True,
            "use_mlock": False,
            "use_mmap": True
        }
        
        print("Initializing model...")
        llm = Llama(**model_config)
        print("Model loaded successfully")
        
    except Exception as e:
        print(f"Model initialization failed: {str(e)}")
        raise RuntimeError("Failed to initialize model") from e

def process_input(job):
    try:
        job_input = job["input"]
        prompt = job_input.get("prompt", "")
        max_tokens = job_input.get("max_tokens", 200)
        temperature = job_input.get("temperature", 0.7)
        
        if not prompt:
            return {"error": "Empty prompt provided"}
            
        output = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        
        return {"response": output["choices"][0]["text"]}
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    initialize_model()
    runpod.serverless.start({"handler": process_input})