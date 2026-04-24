import modal
from fastapi import Request
from fastapi.responses import StreamingResponse

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"  # non-quantized, vLLM handles its own optimization

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.6.3",
        "fastapi[standard]",
        "huggingface_hub",
        "torch==2.4.0",
        "transformers==4.44.2",  # pinned for tokenizer compatibility
        "tokenizers==0.19.1",    # pinned for tokenizer compatibility
    )
    .run_commands(
        f"python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('{BASE_MODEL}')\"",
        secrets=[modal.Secret.from_name("huggingface-secret")]
    )
)

app = modal.App("pangasinan-agent-backend")

@app.cls(
    image=base_image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
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
    async def generate(self, prompt: str, hf_repo_id: str) -> str:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        import uuid

        sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
        request_id = str(uuid.uuid4())
        lora_request = LoRARequest(hf_repo_id, 1, hf_repo_id)

        full_output = ""
        async for request_output in self.engine.generate(
            prompt, sampling_params, request_id, lora_request=lora_request
        ):
            if request_output.outputs:
                full_output = request_output.outputs[0].text

        return full_output

@app.function(image=base_image)
@modal.fastapi_endpoint(method="POST")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    hf_repo_id = data.get("hf_repo_id")

    if not hf_repo_id:
        return {"error": "Missing hf_repo_id"}

    engine = VllmEngine()
    result = await engine.generate.remote.aio(prompt, hf_repo_id=hf_repo_id)

    return {"response": result}