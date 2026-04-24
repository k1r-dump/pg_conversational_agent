import modal
from fastapi import Request
from fastapi.responses import StreamingResponse

# 1. Define the base model
BASE_MODEL = "unsloth/llama-3-8b-Instruct"

# 2. Define the container environment
vllm_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("vllm==0.4.1", "huggingface_hub[cli]") # Ensure HF hub is installed
    .run_commands(
        # Use the new 'hf' command or the modern 'huggingface-cli' flag
        f"hf download {BASE_MODEL}" 
    )
)

# --- THIS IS THE PART THAT WAS MISSING ---
app = modal.App("pangasinan-agent-backend")
# -----------------------------------------

# 3. Define the Inference Engine
@app.cls(
    image=vllm_image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-secret")], 
    # container_idle_timeout=300, <--- DELETE THIS
    scaledown_window=300,        # <--- ADD THIS (renamed as of Feb 2025)
)   
class VllmEngine:
    @modal.enter()
    def start_engine(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        engine_args = AsyncEngineArgs(
            model=BASE_MODEL,
            gpu_memory_utilization=0.90,
            enable_lora=True,          
            max_loras=4,               
            max_model_len=2048,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @modal.method()
    async def generate(self, prompt: str, hf_repo_id: str):
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        import uuid

        sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
        request_id = str(uuid.uuid4())
        
        # Pulls the adapter directly from your private HF repo
        lora_request = LoRARequest(hf_repo_id, 1, hf_repo_id)

        results_generator = self.engine.generate(
            prompt, 
            sampling_params, 
            request_id, 
            lora_request=lora_request
        )

        async for request_output in results_generator:
            if request_output.outputs:
                yield request_output.outputs[0].text

# 4. The REST API Endpoint
@modal.fastapi_endpoint(method="POST")
async def chat_endpoint(request: Request):
    data = await request.json()
    # ... your existing logic
    prompt = data.get("prompt", "")
    hf_repo_id = data.get("hf_repo_id") # e.g. "Kiruu/pangasinan-v1"

    if not hf_repo_id:
        return {"error": "Missing hf_repo_id"}

    engine = VllmEngine()

    async def stream_response():
        previous_text_len = 0
        async for full_text in engine.generate.remote_gen(prompt, hf_repo_id=hf_repo_id):
            new_text = full_text[previous_text_len:]
            previous_text_len = len(full_text)
            yield new_text

    return StreamingResponse(stream_response(), media_type="text/plain")