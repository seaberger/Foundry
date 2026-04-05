"""Activate a candidate adapter for autoresearch evaluation.

Writes the adapter config to the Modal volume, deploys a temporary vLLM
endpoint, waits for health, and prints the endpoint URL (with /v1) to stdout.

After evaluation, stop the endpoint with:
    modal app stop foundry-autoresearch-candidate

Usage:
    python experiments/autoresearch/backend/activate_candidate.py \
        --adapter-name madison-qwen3-probe-20260404-123456 \
        --adapter-path /adapters/experiments/madison-qwen3-probe-20260404-123456
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODAL_APP_NAME = "foundry-autoresearch-candidate"
ENDPOINT_BASE = "https://seaberger--foundry-autoresearch-candidate-serve.modal.run"
SERVE_SCRIPT = Path(__file__).resolve().parent / "serve_candidate.py"
HEALTH_TIMEOUT = 600  # 10 minutes


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy candidate adapter for evaluation")
    parser.add_argument("--adapter-name", required=True)
    parser.add_argument("--adapter-path", required=True)
    args = parser.parse_args()

    # Step 1: Write config to volume via Modal function
    print(f"Writing adapter config: {args.adapter_name}", file=sys.stderr)
    proc = subprocess.run(
        [
            "modal", "run", str(SERVE_SCRIPT),
            "--adapter-name", args.adapter_name,
            "--adapter-path", args.adapter_path,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"Failed to write config:\n{proc.stderr}", file=sys.stderr)
        return 1

    # Step 2: Stop any existing deployment (forces fresh cold start with new config)
    print("Stopping existing candidate endpoint...", file=sys.stderr)
    subprocess.run(
        ["modal", "app", "stop", MODAL_APP_NAME],
        capture_output=True,
        check=False,
    )

    # Step 3: Deploy fresh
    print("Deploying candidate endpoint...", file=sys.stderr)
    proc = subprocess.run(
        ["modal", "deploy", str(SERVE_SCRIPT)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"Deploy failed:\n{proc.stderr}", file=sys.stderr)
        return 1

    # Step 4: Poll health until vLLM is ready
    print("Waiting for endpoint health...", file=sys.stderr)
    deadline = time.time() + HEALTH_TIMEOUT
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{ENDPOINT_BASE}/health", method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    # Print endpoint URL (with /v1) to stdout for capture by train.py
                    print(f"{ENDPOINT_BASE}/v1")
                    return 0
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass
        time.sleep(10)

    print("Timeout waiting for candidate endpoint", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
