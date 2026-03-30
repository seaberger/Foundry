"""Convert merged Qwen 3-32B model to GGUF Q4_K_M format on Modal.

Steps:
  1. Clone latest llama.cpp
  2. Convert HF model to F16 GGUF
  3. Quantize to Q4_K_M
  4. Save GGUF to Modal volume for download

Usage:
    modal run modal_convert_gguf_qwen.py
    modal run modal_convert_gguf_qwen.py --quant Q5_K_M
    modal run modal_convert_gguf_qwen.py --model-name madison-qwen3-val-v1
"""

from __future__ import annotations

import modal

MINUTES = 60

app = modal.App("foundry-qwen3-gguf")

adapter_vol = modal.Volume.from_name("foundry-adapters", create_if_missing=True)
model_cache_vol = modal.Volume.from_name("foundry-model-cache", create_if_missing=True)

convert_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "cmake", "build-essential")
    .run_commands(
        "git clone https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -B build && cmake --build build --config Release -j$(nproc)",
    )
    .uv_pip_install(
        "numpy",
        "sentencepiece",
        "transformers>=4.54.0",
        "huggingface_hub",
        "gguf",
        "torch",
        "safetensors",
        "protobuf",
    )
    .env({"HF_HOME": "/model_cache"})
)


@app.function(
    image=convert_image,
    volumes={
        "/adapters": adapter_vol,
        "/model_cache": model_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=65536,
    timeout=60 * MINUTES,
    cpu=8,
)
def convert_and_quantize(
    merged_model_path: str = "/adapters/merged/madison-qwen3-val-v1-16bit",
    output_dir: str = "/adapters/gguf",
    model_name: str = "madison-qwen3-val-v1",
    quant: str = "Q4_K_M",
):
    """Convert merged HF model to GGUF and quantize."""
    import subprocess
    from pathlib import Path

    model_dir = Path(merged_model_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json at {model_dir}")
    print(f"Model found at {model_dir}")
    for f in sorted(model_dir.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name} ({size_mb:.1f} MB)")

    # Qwen 3 doesn't need the tokenizer.model workaround that Gemma 3 required

    # Convert to F16 GGUF
    f16_gguf = out_dir / f"{model_name}-f16.gguf"
    print(f"\nConverting to F16 GGUF: {f16_gguf}")

    result = subprocess.run(
        [
            "python3", "/opt/llama.cpp/convert_hf_to_gguf.py",
            str(model_dir),
            "--outfile", str(f16_gguf),
            "--outtype", "f16",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"convert_hf_to_gguf.py failed with code {result.returncode}")

    f16_size = f16_gguf.stat().st_size / 1024 / 1024 / 1024
    print(f"F16 GGUF created: {f16_size:.1f} GB")

    # Quantize
    quant_gguf = out_dir / f"{model_name}-{quant.lower()}.gguf"
    print(f"\nQuantizing to {quant}: {quant_gguf}")

    result = subprocess.run(
        [
            "/opt/llama.cpp/build/bin/llama-quantize",
            str(f16_gguf),
            str(quant_gguf),
            quant,
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"llama-quantize failed with code {result.returncode}")

    quant_size = quant_gguf.stat().st_size / 1024 / 1024 / 1024
    print(f"\n{quant} GGUF created: {quant_size:.1f} GB")

    # Clean up F16 GGUF
    print(f"\nRemoving F16 GGUF to save volume space...")
    f16_gguf.unlink()

    adapter_vol.commit()

    print(f"\n{'='*60}")
    print(f"GGUF conversion complete!")
    print(f"  Quantized model: {quant_gguf}")
    print(f"  Size: {quant_size:.1f} GB")
    print(f"  Quant: {quant}")
    print(f"\nTo download:")
    print(f"  modal volume get foundry-adapters gguf/{quant_gguf.name} .")
    print(f"{'='*60}")

    return {
        "path": str(quant_gguf),
        "size_gb": round(quant_size, 1),
        "quant": quant,
    }


@app.local_entrypoint()
def main(
    quant: str = "Q4_K_M",
    model_name: str = "madison-qwen3-val-v1",
    merged_path: str = "",
):
    if not merged_path:
        merged_path = f"/adapters/merged/{model_name}-16bit"
    result = convert_and_quantize.remote(
        merged_model_path=merged_path,
        model_name=model_name,
        quant=quant,
    )
    print(f"\nResult: {result}")
