"""Serve fine-tuned Madison model on Modal for evaluation.

Loads the ORPO-trained LoRA adapter on top of Gemma 3 27B and exposes
an OpenAI-compatible /v1/chat/completions endpoint.

Usage:
    # Start serving (keeps running until stopped)
    modal serve modal_serve_madison.py

    # Then run eval from local machine:
    cd ~/Repositories/Foundry
    .venv/bin/python -m foundry.press.evaluate \
        --endpoint https://seaberger--foundry-madison-serve-generate.modal.run/v1 \
        --model madison-orpo-v3b \
        --tag orpo-v3b
"""

from __future__ import annotations

import modal

MINUTES = 60
GPU = "A100"

app = modal.App("foundry-madison-serve")

model_cache_vol = modal.Volume.from_name("foundry-model-cache", create_if_missing=True)
adapter_vol = modal.Volume.from_name("foundry-adapters", create_if_missing=True)

serving_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.9.0",
        "huggingface_hub==0.34.2",
        "peft==0.16.0",
        "transformers==4.54.0",
        "unsloth[cu128-torch270]==2025.7.8",
        "unsloth_zoo==2025.7.10",
    )
    .env({"HF_HOME": "/model_cache"})
)

with serving_image.imports():
    import unsloth  # noqa: F401,I001
    from unsloth import FastLanguageModel


@app.cls(
    image=serving_image,
    gpu=GPU,
    timeout=60 * MINUTES,
    volumes={
        "/model_cache": model_cache_vol,
        "/adapters": adapter_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    container_idle_timeout=10 * MINUTES,
)
class MadisonModel:
    adapter_name: str = "madison-orpo-v3b-lr2e5"
    base_model: str = "google/gemma-3-27b-it"
    max_seq_length: int = 2048

    @modal.enter()
    def load_model(self):
        """Load base model + LoRA adapter on container start."""
        from peft import PeftModel

        print(f"Loading {self.base_model}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        adapter_path = f"/adapters/experiments/{self.adapter_name}"
        print(f"Loading adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        FastLanguageModel.for_inference(self.model)
        print("Model ready for inference.")

    @modal.web_endpoint(method="POST")
    def v1_chat_completions(self, request: dict):
        """OpenAI-compatible /v1/chat/completions endpoint."""
        import json
        import time

        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 1024)
        temperature = request.get("temperature", 0.7)

        # Build prompt from messages
        conversation = []
        for msg in messages:
            conversation.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        inputs = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        start = time.time()
        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                use_cache=True,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs.shape[1]:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        elapsed = time.time() - start

        # Return OpenAI-compatible response
        return {
            "id": f"chatcmpl-madison-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", self.adapter_name),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": inputs.shape[1],
                "completion_tokens": len(new_tokens),
                "total_tokens": inputs.shape[1] + len(new_tokens),
            },
        }
