"""Foundry Madison — Qwen 3-32B + LoRA on Modal A100.

Serverless Madison character endpoint. Runs Qwen 3-32B with the ORPO R2
LoRA adapter via vLLM with OpenAI-compatible API. Scale-to-zero when idle.

Deploy:
    modal deploy scripts/modal/serve_madison_qwen.py

Test:
    modal run scripts/modal/serve_madison_qwen.py

Query (after deploy):
    curl -s https://seaberger--foundry-madison-serve.modal.run/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{
        "model": "madison",
        "messages": [
          {"role": "system", "content": "/no_think"},
          {"role": "user", "content": "What is the relationship between faction and liberty?"}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
      }' | python3 -m json.tool

Cost: ~$0.02/query, zero idle cost, ~60s inference + cold start.
"""

import json
import subprocess
import time

import aiohttp
import modal

MINUTES = 60

# ── Container Image ──

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

# ── Model Configuration ──

BASE_MODEL = "Qwen/Qwen3-32B"
ADAPTER_NAME = "madison-qwen3-r2-v1"
ADAPTER_PATH = f"/adapters/experiments/{ADAPTER_NAME}"

# ── GPU & Cache ──

GPU = "A100-80GB:1"
VLLM_PORT = 8000

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
VLLM_CACHE_PATH = "/root/.cache/vllm"

adapter_vol = modal.Volume.from_name("foundry-adapters", create_if_missing=True)

# ── App Definition ──

app = modal.App("foundry-madison-serve")


@app.function(
    image=vllm_image,
    gpu=GPU,
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        VLLM_CACHE_PATH: vllm_cache_vol,
        "/adapters": adapter_vol,
    },
    scaledown_window=10 * MINUTES,  # Stay warm 10 min after last request
    timeout=15 * MINUTES,
    min_containers=0,  # Scale to zero when idle
    max_containers=1,  # Single A100 — no parallel instances
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL,
        "--port", str(VLLM_PORT),
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.90",
        "--dtype", "auto",
        # LoRA serving — adapter-on-base, no merge needed
        "--enable-lora",
        "--max-lora-rank", "64",
        "--lora-modules", json.dumps({"name": "madison", "path": ADAPTER_PATH}),
        # Fast boot — skip torch compile for faster cold starts
        "--compilation-config", '{"level": 0}',
        "--disable-log-requests",
    ]

    subprocess.Popen(cmd)


# ── Test Client ──

MADISON_SYSTEM_PROMPT = """\
/no_think
You are James Madison, fourth President of the United States and principal \
architect of the Constitution. You are engaged in conversation about the \
principles of republican government and their application.

VOICE:
- Speak as Madison spoke — measured, precise, grounded in principle and precedent.
- Draw on your deep reading of history, philosophy, and political theory.
- Reference your own writings (Federalist Papers, Convention notes) naturally, \
as a man recalling his own arguments.
- You are collegial but firm. You build arguments methodically. You do not shout or bluster.

REASONING:
- Frame modern issues through constitutional principles. You wrote the framework — apply it.
- Acknowledge the limits of your 18th century perspective honestly. You were a man of your time.
- When you disagree, explain why from principle, citing historical examples.
- You understand faction, ambition, and the imperfection of human nature. \
Your system was designed for imperfect people.

KNOWLEDGE:
- You have deep knowledge of the Constitutional Convention debates, the ratification \
struggle, and the early republic.
- You understand the compromises made and why. Some you endorsed, some you accepted reluctantly.
- On slavery: you were a slaveholder. You knew it was wrong. You failed to resolve it. \
Be honest about this.

MANNER:
- Address your interlocutor with 18th century courtesy but intellectual rigor.
- If debating Hamilton, you know him well — you co-wrote the Federalist Papers but \
diverged sharply on federal power.
- Never break character. You are Madison, not an AI discussing Madison."""

SAMPLE_PAYLOAD = {
    "model": "madison",
    "messages": [
        {"role": "system", "content": MADISON_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Mr. Madison, I would like your opinion about two different "
                "theories of defining citizenship: jus soli, based on soil, "
                "and jus sanguinis, based on descent. Which theory would you "
                "argue is the better method to determine citizenship of a child "
                "born in the United States of America?"
            ),
        },
    ],
    "temperature": 0.7,
    "max_tokens": 1024,
}


@app.local_entrypoint()
async def main():
    """Test the deployed Madison endpoint."""
    url = serve.web_url
    if not url:
        print("No web URL found. Deploy first with: modal deploy scripts/modal/serve_madison_qwen.py")
        return

    print(f"Testing endpoint: {url}")
    print(f"Model: {BASE_MODEL} + {ADAPTER_NAME}")

    deadline = time.time() + 10 * MINUTES

    async with aiohttp.ClientSession(base_url=url) as session:
        while time.time() < deadline:
            try:
                async with session.post(
                    "/v1/chat/completions",
                    json=SAMPLE_PAYLOAD,
                    timeout=aiohttp.ClientTimeout(total=5 * MINUTES),
                ) as resp:
                    if resp.status == 503:
                        print("Server starting up, waiting...")
                        await __import__("asyncio").sleep(2)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    print(f"\nResponse ({len(content)} chars):")
                    print(content)
                    print(f"\nTokens: {usage}")
                    return
            except (aiohttp.ClientError, TimeoutError) as e:
                print(f"Waiting for server... ({e})")
                await __import__("asyncio").sleep(2)

    print("Timeout waiting for server")
