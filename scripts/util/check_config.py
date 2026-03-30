"""Dump the full config.json from the merged model."""
import modal

app = modal.App("check-config")
vol = modal.Volume.from_name("foundry-adapters")

@app.function(volumes={"/adapters": vol}, timeout=60)
def dump_config():
    import json
    from pathlib import Path
    model_dir = Path("/adapters/merged/madison-orpo-v3b-lr2e5-16bit")
    config = json.loads((model_dir / "config.json").read_text())
    print(json.dumps(config, indent=2))

@app.local_entrypoint()
def main():
    dump_config.remote()
