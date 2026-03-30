"""Patch tokenizer_config.json on Modal volume to add image_token_id for vLLM.

The merged model's tokenizer was saved from base Gemma 3 text tokenizer
which doesn't include the image_token_id needed by Gemma3Processor.
The model config already has image_token_index=262144 — we just need
the tokenizer to know about it too.
"""
import modal

app = modal.App("patch-tokenizer")
vol = modal.Volume.from_name("foundry-adapters")

@app.function(volumes={"/adapters": vol}, timeout=120)
def patch_tokenizer():
    import json
    from pathlib import Path

    model_dir = Path("/adapters/merged/madison-orpo-v3b-lr2e5-16bit")
    tc_path = model_dir / "tokenizer_config.json"

    tc = json.loads(tc_path.read_text())
    print(f"Before patch - keys: {list(tc.keys())[:20]}")
    print(f"Has image_token: {'image_token' in tc}")

    # Add image token config matching the model config's image_token_index
    # From config.json: "image_token_index": 262144
    if "image_token" not in tc:
        # The special token for images in Gemma 3 VLM
        tc["image_token"] = "<image_soft_token>"
        tc["image_token_id"] = 262144

        # Also check if added_tokens_decoder has this token
        atd = tc.get("added_tokens_decoder", {})
        if "262144" not in atd:
            atd["262144"] = {
                "content": "<image_soft_token>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }
            tc["added_tokens_decoder"] = atd

        tc_path.write_text(json.dumps(tc, indent=2))
        vol.commit()
        print("Patched tokenizer_config.json with image_token_id=262144")
    else:
        print(f"Already has image_token: {tc['image_token']}")

    # Also check preprocessor_config.json
    pp_path = model_dir / "preprocessor_config.json"
    if pp_path.exists():
        pp = json.loads(pp_path.read_text())
        print(f"\npreprocessor_config.json: {json.dumps(pp, indent=2)[:500]}")
    else:
        print("\nNo preprocessor_config.json found")

    # Verify
    tc2 = json.loads(tc_path.read_text())
    print(f"\nAfter patch - image_token: {tc2.get('image_token')}")
    print(f"After patch - image_token_id: {tc2.get('image_token_id')}")

@app.local_entrypoint()
def main():
    patch_tokenizer.remote()
