"""Quick check of merged model weight key format on Modal."""
import modal

app = modal.App("check-weights")
vol = modal.Volume.from_name("foundry-adapters")

@app.function(volumes={"/adapters": vol}, timeout=60)
def check_weight_format():
    import json
    from pathlib import Path

    model_dir = Path("/adapters/merged/madison-orpo-v3b-lr2e5-16bit")

    # Check config.json
    config = json.loads((model_dir / "config.json").read_text())
    print("=== config.json ===")
    print(f"architectures: {config.get('architectures')}")
    print(f"model_type: {config.get('model_type')}")
    print(f"Has text_config: {'text_config' in config}")
    if 'rope_scaling' in config:
        print(f"rope_scaling keys: {list(config['rope_scaling'].keys()) if isinstance(config['rope_scaling'], dict) else config['rope_scaling']}")
    if 'rope_parameters' in config:
        print("rope_parameters: present")
    if 'text_config' in config:
        tc = config['text_config']
        print(f"text_config keys: {list(tc.keys())[:10]}")

    # Check safetensors index
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        sample_keys = sorted(weight_map.keys())[:15]
        print(f"\n=== Weight keys (first 15) ===")
        for k in sample_keys:
            print(f"  {k}")
        has_lm_prefix = any(k.startswith("language_model.") for k in weight_map)
        print(f"\nHas 'language_model.' prefix: {has_lm_prefix}")
        print(f"Total weight keys: {len(weight_map)}")

    # List files
    print(f"\n=== Files ===")
    for f in sorted(model_dir.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name} ({size_mb:.1f} MB)")

@app.local_entrypoint()
def main():
    check_weight_format.remote()
