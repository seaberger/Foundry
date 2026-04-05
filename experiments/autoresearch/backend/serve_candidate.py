"""Ephemeral vLLM endpoint for autoresearch candidate evaluation.

Serves Qwen 3-32B + a candidate LoRA adapter on a single A100-80GB.
Reads adapter config from a JSON file on the foundry-adapters volume,
written by activate_candidate.py before deployment.

This endpoint is short-lived: deployed for evaluation, then stopped.

Deploy (called by activate_candidate.py):
    modal deploy experiments/autoresearch/backend/serve_candidate.py

Write config:
    modal run experiments/autoresearch/backend/serve_candidate.py \
        --adapter-name madison-qwen3-probe-20260404-123456 \
        --adapter-path /adapters/experiments/madison-qwen3-probe-20260404-123456
"""

import json
import subprocess
from pathlib import Path

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
# Fixed model name matching prepare.py's FOUNDRY_AUTORESEARCH_MODEL_NAME default
SERVE_MODEL_NAME = "madison-qwen3-probe"
ADAPTER_CONFIG_PATH = "/adapters/autoresearch-active.json"

# ── GPU & Cache ──

GPU = "A100-80GB:1"
VLLM_PORT = 8000

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
VLLM_CACHE_PATH = "/root/.cache/vllm"

adapter_vol = modal.Volume.from_name("foundry-adapters", create_if_missing=True)

# ── App Definition ──

app = modal.App("foundry-autoresearch-candidate")


@app.function(
    image=vllm_image,
    gpu=GPU,
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        VLLM_CACHE_PATH: vllm_cache_vol,
        "/adapters": adapter_vol,
    },
    scaledown_window=5 * MINUTES,   # Short — ephemeral endpoint
    timeout=15 * MINUTES,
    min_containers=0,
    max_containers=1,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    """Start vLLM with the candidate adapter from the config file."""
    config_path = Path(ADAPTER_CONFIG_PATH)
    if not config_path.exists():
        raise FileNotFoundError(
            f"No adapter config at {ADAPTER_CONFIG_PATH}. "
            "Run activate_candidate.py first."
        )

    config = json.loads(config_path.read_text())
    adapter_path = config["adapter_path"]

    print(f"Serving candidate adapter: {adapter_path}")
    print(f"Model name: {SERVE_MODEL_NAME}")

    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL,
        "--port", str(VLLM_PORT),
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.90",
        "--dtype", "auto",
        # LoRA serving
        "--enable-lora",
        "--max-lora-rank", "64",
        "--lora-modules", json.dumps({"name": SERVE_MODEL_NAME, "path": adapter_path}),
        # Fast boot
        "--compilation-config", '{"level": 0}',
        "--disable-log-requests",
    ]

    subprocess.Popen(cmd)


# ── Config Writer ──

@app.function(
    volumes={"/adapters": adapter_vol},
)
def write_config(adapter_name: str, adapter_path: str):
    """Write the candidate adapter config to the shared volume."""
    config = {
        "adapter_name": adapter_name,
        "adapter_path": adapter_path,
    }
    Path(ADAPTER_CONFIG_PATH).write_text(json.dumps(config, indent=2))
    adapter_vol.commit()
    print(f"Config written: {adapter_name} -> {adapter_path}")


@app.local_entrypoint()
def main(adapter_name: str = "", adapter_path: str = ""):
    """Write adapter config to the volume."""
    if adapter_name and adapter_path:
        write_config.remote(adapter_name, adapter_path)
    else:
        print("Usage: modal run serve_candidate.py --adapter-name NAME --adapter-path PATH")
