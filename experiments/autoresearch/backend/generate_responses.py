"""One-shot response generation on Modal — no web server needed.

Loads Qwen 3-32B + LoRA adapter, generates responses for all prompts,
prints them as JSON lines between RESPONSES_START/END markers.

Usage:
    modal run experiments/autoresearch/backend/generate_responses.py \
        --adapter-path /adapters/experiments/madison-qwen3-dryrun-20260404-202618/checkpoint-150 \
        --prompts-file /tmp/eval_prompts.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

MINUTES = 60

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

BASE_MODEL = "Qwen/Qwen3-32B"
MODEL_NAME = "madison-qwen3-probe"
GPU = "A100-80GB:1"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
adapter_vol = modal.Volume.from_name("foundry-adapters", create_if_missing=True)

app = modal.App("foundry-autoresearch-generate")


@app.function(
    image=vllm_image,
    gpu=GPU,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/adapters": adapter_vol,
    },
    timeout=30 * MINUTES,
)
def generate_all(adapter_path: str, prompts_json: str) -> str:
    """Load model + adapter, generate responses for all prompts."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    prompts = [json.loads(line) for line in prompts_json.strip().split("\n") if line.strip()]
    print(f"Loaded {len(prompts)} prompts")
    print(f"Adapter: {adapter_path}")

    # Check adapter exists
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    print(f"Adapter files: {list(adapter_dir.iterdir())}")

    # Load model
    llm = LLM(
        model=BASE_MODEL,
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        enable_lora=True,
        max_lora_rank=64,
        dtype="auto",
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
    )

    lora_request = LoRARequest(
        lora_name=MODEL_NAME,
        lora_int_id=1,
        lora_path=adapter_path,
    )

    # Build prompts with Madison system prompt
    system_prompt = (
        "You are James Madison, fourth President of the United States. "
        "Respond in character as Madison would — measured, precise, grounded in principle and precedent. "
        "Never break character."
    )

    # Generate all responses
    results = []
    for i, p in enumerate(prompts):
        prompt_text = p["prompt"]

        # Format as chat messages
        messages = [
            {"role": "system", "content": f"/no_think\n{system_prompt}"},
            {"role": "user", "content": prompt_text},
        ]

        # Apply chat template
        tokenizer = llm.get_tokenizer()
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        print(f"[{i+1}/{len(prompts)}] Generating for {p['id']}...")

        outputs = llm.generate(
            [formatted],
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

        response_text = outputs[0].outputs[0].text
        # Strip any residual <think> tags
        if "<think>" in response_text:
            parts = response_text.split("</think>")
            if len(parts) > 1:
                response_text = parts[-1].strip()

        results.append({
            "id": p["id"],
            "category": p["category"],
            "prompt": prompt_text,
            "response": response_text,
        })

        print(f"  Response length: {len(response_text)} chars")

    # Return as JSON lines
    output_lines = [json.dumps(r) for r in results]
    return "\n".join(output_lines)


@app.function(
    volumes={"/adapters": adapter_vol},
)
def upload_prompts(data: bytes) -> str:
    """Upload prompts file to the adapters volume."""
    path = Path("/adapters/data/eval_prompts.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    adapter_vol.commit()
    return str(path)


@app.local_entrypoint()
def main(
    adapter_path: str = "/adapters/experiments/madison-qwen3-dryrun-20260404-202618/checkpoint-150",
    prompts_file: str = "/tmp/eval_prompts.jsonl",
):
    # Read prompts locally
    prompts_json = Path(prompts_file).read_text()

    # Generate responses on GPU
    responses_json = generate_all.remote(adapter_path, prompts_json)

    # Output with markers for parsing
    print("RESPONSES_START")
    print(responses_json)
    print("RESPONSES_END")
