"""Foundry Chamber — Madison chatbot on Modal.

Two-tier architecture:
  Gateway (CPU, ~1s start) → Loading page → Chamber (A100, ~2 min cold start)

Deploy:
    modal deploy scripts/modal/serve_chamber.py

User-facing URL (gateway — always instant):
    https://seaberger--foundry-chamber-gateway-web.modal.run

Export eval logs:
    curl https://seaberger--foundry-chamber-chamber-web.modal.run/api/export > evals.jsonl
    # or: modal volume get foundry-chamber-db chamber-evals.jsonl

Closes: https://github.com/seaberger/Foundry/issues/1
"""

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid

import modal

MINUTES = 60
APP_DIR = "/app"
EVALS_PATH = "/data/chamber-evals.jsonl"
log = logging.getLogger("chamber.modal")

# ── Container Images ──

gateway_image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi>=0.115")

chamber_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.19.0",
        "huggingface-hub==0.36.0",
        "fastapi>=0.115",
        "uvicorn[standard]>=0.34",
        "jinja2>=3.1",
        "python-multipart>=0.0.18",
        "sse-starlette>=2.0",
        "pydantic>=2.0",
        "pyyaml>=6.0",
        "httpx>=0.27",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir("src", remote_path=f"{APP_DIR}/src", copy=True)
    .add_local_dir("templates", remote_path=f"{APP_DIR}/templates", copy=True)
    .add_local_dir("static", remote_path=f"{APP_DIR}/static", copy=True)
    .add_local_dir("config", remote_path=f"{APP_DIR}/config", copy=True)
)

# ── Model Configuration ──

BASE_MODEL = "Qwen/Qwen3-32B"
MERGED_MODEL_PATH = "/merged/madison-qwen3-r2-v1-merged"
VLLM_PORT = 8000

# ── Volumes ──

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
adapter_vol = modal.Volume.from_name("foundry-adapters", create_if_missing=True)
merged_vol = modal.Volume.from_name("foundry-merged-models", create_if_missing=True)
chamber_db_vol = modal.Volume.from_name("foundry-chamber-db", create_if_missing=True)

# ── App ──

app = modal.App("foundry-chamber")

# ── vLLM readiness ──

_vllm_ready = threading.Event()


def _poll_vllm_ready():
    import httpx

    url = f"http://localhost:{VLLM_PORT}/health"
    while True:
        try:
            resp = httpx.get(url, timeout=5)
            if resp.status_code == 200:
                log.info("vLLM is ready")
                _vllm_ready.set()
                return
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(2)


def _strip_think_tags(text):
    """Remove Qwen3 <think>...</think> blocks from output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def _append_eval_log(session_id, character, system_prompt, user_msg, assistant_msg):
    """Append a conversation turn to the JSONL eval log on the volume."""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id": session_id,
        "character": character,
        "model": f"{BASE_MODEL}+{ADAPTER_NAME}",
        "system_prompt_length": len(system_prompt),
        "user": user_msg,
        "assistant": assistant_msg,
    }
    try:
        with open(EVALS_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        log.error("Failed to write eval log: %s", e)


# ════════════════════════════════════════════════════════════════
# TIER 1: Gateway — CPU-only, starts in ~1 second
# ════════════════════════════════════════════════════════════════

CHAMBER_URL = "https://seaberger--foundry-chamber-chamber-web.modal.run"

LOADING_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>The Foundry — Loading</title>
    <link rel="icon" href="CHAMBER_URL_PLACEHOLDER/static/images/foundry-logo.jpg">
    <meta property="og:image" content="CHAMBER_URL_PLACEHOLDER/static/images/foundry-logo.jpg">
    <meta property="og:title" content="The Foundry — Conversations with America's Founders">
    <meta property="og:description" content="Chat with a fine-tuned AI James Madison. Ask about the Constitution, the Federalist Papers, faction, liberty, and the founding of the republic.">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://seaberger--foundry-chamber-gateway-web.modal.run">
    <meta name="description" content="Chat with a fine-tuned AI James Madison. Ask about the Constitution, the Federalist Papers, faction, liberty, and the founding of the republic.">
    <style>
        :root {
            --parchment: #f5f0e8;
            --parchment-dark: #e8e0d0;
            --ink: #2c2416;
            --ink-light: #5a4e3c;
            --accent: #8b6914;
            --border: #d4c9b5;
            --font-body: Georgia, 'Times New Roman', serif;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: var(--font-body);
            background: var(--parchment);
            color: var(--ink);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .site-header {
            position: fixed; top: 0; width: 100%;
            display: flex; align-items: baseline; gap: 1rem;
            padding: 1rem 2rem;
            border-bottom: 2px solid var(--border);
            background: var(--parchment-dark);
        }
        .site-title {
            font-size: 1.5rem; font-weight: bold;
            color: var(--ink); text-decoration: none;
            letter-spacing: 0.02em;
        }
        .site-subtitle {
            font-size: 0.9rem; color: var(--ink-light);
            font-style: italic;
        }
        .loading-container {
            text-align: center; padding: 2rem;
            max-width: 500px;
        }
        .loading-title {
            font-size: 1.6rem; margin-bottom: 1.5rem;
            letter-spacing: 0.02em;
        }
        .loading-message {
            font-size: 1.1rem; line-height: 1.6;
            color: var(--ink-light);
            margin-bottom: 2rem;
            font-style: italic;
        }
        .quill {
            font-size: 2.5rem;
            animation: write 2s ease-in-out infinite;
            display: inline-block;
            margin-bottom: 1.5rem;
        }
        @keyframes write {
            0%, 100% { transform: rotate(-5deg) translateY(0); }
            25% { transform: rotate(5deg) translateY(-4px); }
            50% { transform: rotate(-3deg) translateY(0); }
            75% { transform: rotate(4deg) translateY(-2px); }
        }
        .status {
            font-size: 0.85rem;
            color: var(--ink-light);
            margin-top: 1rem;
        }
        .dots::after {
            content: '';
            animation: dots 1.5s steps(4, end) infinite;
        }
        @keyframes dots {
            0% { content: ''; }
            25% { content: '.'; }
            50% { content: '..'; }
            75% { content: '...'; }
        }
        .error { color: #8b1414; margin-top: 1rem; display: none; }
    </style>
</head>
<body>
    <header class="site-header">
        <span class="site-title">The Foundry</span>
        <span class="site-subtitle">Conversations with America's Founders</span>
    </header>

    <div class="loading-container">
        <div class="quill">&#x1F4DC;</div>
        <h1 class="loading-title">Preparing the Chamber</h1>
        <p class="loading-message">
            Mr. Madison is reviewing his notes and preparing to receive visitors.
            This may take a moment if the chamber has been idle.
        </p>
        <p class="status">Loading Madison LLM Model<span class="dots"></span></p>
        <p class="error" id="error"></p>
    </div>

    <script>
        const CHAMBER = "CHAMBER_URL_PLACEHOLDER";
        let attempts = 0;
        let readyCount = 0;

        async function checkHealth() {
            attempts++;
            try {
                const resp = await fetch("/api/chamber-status", { signal: AbortSignal.timeout(15000) });
                const data = await resp.json();
                if (data.vllm_ready) {
                    readyCount++;
                    // Require 2 consecutive ready signals to avoid race conditions
                    if (readyCount >= 2) {
                        document.querySelector('.status').textContent =
                            'Madison is ready — entering the Chamber...';
                        setTimeout(() => { window.location.href = CHAMBER; }, 1000);
                        return;
                    }
                    document.querySelector('.status').innerHTML =
                        'Model loaded, performing final preparations<span class="dots"></span>';
                } else {
                    readyCount = 0;
                    if (data.status === 'loading') {
                        document.querySelector('.status').innerHTML =
                            'Model loaded, warming up<span class="dots"></span>';
                    } else {
                        document.querySelector('.status').innerHTML =
                            'Waking the Chamber from cold storage<span class="dots"></span>';
                    }
                }
            } catch (e) {
                readyCount = 0;
                if (attempts > 90) {
                    document.getElementById('error').style.display = 'block';
                    document.getElementById('error').textContent =
                        'The chamber is taking longer than expected. Please refresh the page.';
                }
            }
            setTimeout(checkHealth, 3000);
        }

        checkHealth();
    </script>
</body>
</html>""".replace("CHAMBER_URL_PLACEHOLDER", CHAMBER_URL)


@app.cls(image=gateway_image)
class Gateway:
    @modal.asgi_app()
    def web(self):
        import urllib.error
        import urllib.request

        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse

        gw = FastAPI()

        @gw.get("/", response_class=HTMLResponse)
        async def loading_page():
            return HTMLResponse(LOADING_PAGE)

        @gw.get("/health")
        async def health():
            return {"status": "ok", "role": "gateway"}

        @gw.get("/api/chamber-status")
        async def chamber_status():
            try:
                req = urllib.request.Request(
                    f"{CHAMBER_URL}/health",
                    headers={"User-Agent": "foundry-gateway"},
                )
                resp = urllib.request.urlopen(req, timeout=10)
                data = json.loads(resp.read())
                return JSONResponse(data)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
                return JSONResponse(
                    {"status": "starting", "vllm_ready": False},
                    status_code=202,
                )

        return gw


# ════════════════════════════════════════════════════════════════
# TIER 2: Chamber — A100 GPU, vLLM + FastAPI chatbot
# ════════════════════════════════════════════════════════════════


@app.function(
    image=chamber_image,
    gpu="A100-80GB:1",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/adapters": adapter_vol,
        "/merged": merged_vol,
        "/data": chamber_db_vol,
    },
    scaledown_window=10 * MINUTES,
    timeout=15 * MINUTES,
    min_containers=0,
    max_containers=1,
)
@modal.asgi_app()
def chamber_web():
    """Called once at container startup. Returns the ASGI app."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    sys.path.insert(0, f"{APP_DIR}/src")
    os.chdir(APP_DIR)

    # ── Start vLLM ──
    log.info("Starting vLLM subprocess...")
    vllm_cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MERGED_MODEL_PATH,
        "--tokenizer", BASE_MODEL,
        "--served-model-name", "madison",
        "--port", str(VLLM_PORT),
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.90",
        "--dtype", "auto",
    ]
    subprocess.Popen(vllm_cmd)
    threading.Thread(target=_poll_vllm_ready, daemon=True).start()

    # ── Configure foundry ──
    log.info("Configuring foundry...")
    from foundry import config as config_module
    from foundry.config import FoundryConfig, InferenceConfig, SamplerConfig, StorageConfig

    config_module._config = FoundryConfig(
        inference=InferenceConfig(
            backend="local",
            local_endpoint=f"http://localhost:{VLLM_PORT}/v1",
            timeout=300,
        ),
        storage=StorageConfig(db_path="/data/foundry.db"),
        samplers=SamplerConfig(temperature=0.7, max_tokens=1024),
    )

    from foundry.db import init_db
    init_db()
    log.info("Chamber startup complete — building ASGI app")

    # ── Build FastAPI app ──
    import httpx
    from fastapi import FastAPI, Form, Request
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from sse_starlette.sse import EventSourceResponse

    from foundry.characters.loader import list_characters, load_character
    from foundry.config import get_config
    from foundry.db import get_db, now_iso

    chamber = FastAPI(title="The Foundry Project")
    templates = Jinja2Templates(directory=f"{APP_DIR}/templates")
    chamber.mount("/static", StaticFiles(directory=f"{APP_DIR}/static"), name="static")

    @chamber.get("/health")
    async def health():
        ready = _vllm_ready.is_set()
        return JSONResponse(
            {"status": "ok" if ready else "loading", "vllm_ready": ready},
            status_code=200 if ready else 503,
        )

    @chamber.get("/api/export")
    async def export_evals():
        """Download the JSONL eval log."""
        from fastapi.responses import FileResponse
        from pathlib import Path

        path = Path(EVALS_PATH)
        if not path.exists():
            return JSONResponse({"error": "No eval logs yet"}, status_code=404)
        return FileResponse(
            path,
            media_type="application/x-ndjson",
            filename="chamber-evals.jsonl",
        )

    GATEWAY_URL = "https://seaberger--foundry-chamber-gateway-web.modal.run"

    @chamber.get("/", response_class=HTMLResponse)
    async def sessions_page(request: Request):
        # If someone hits the Chamber directly before model is ready,
        # redirect them to the gateway loading page
        if not _vllm_ready.is_set():
            return RedirectResponse(url=GATEWAY_URL)

        characters = list_characters()
        with get_db() as db:
            rows = db.execute(
                "SELECT id, name, character_ids, last_active, turn_count "
                "FROM sessions ORDER BY last_active DESC"
            ).fetchall()
        sessions = []
        for row in rows:
            char_ids = json.loads(row["character_ids"])
            sessions.append({
                "id": row["id"],
                "name": row["name"],
                "character": char_ids[0] if char_ids else "unknown",
                "last_active": row["last_active"],
                "turn_count": row["turn_count"],
            })

        # Filter characters: only Madison is available, Hamilton is coming soon
        available = [c for c in characters if c == "madison"]
        coming_soon = [c for c in characters if c != "madison"]

        return templates.TemplateResponse(
            "sessions.html",
            {
                "request": request,
                "sessions": sessions,
                "characters": available,
                "coming_soon": coming_soon,
            },
        )

    @chamber.post("/sessions")
    async def create_session(character: str = Form(...), name: str = Form("")):
        if not _vllm_ready.is_set():
            return RedirectResponse(url=GATEWAY_URL)
        card = load_character(character)
        session_id = uuid.uuid4().hex[:12]
        # Prepend /no_think to suppress Qwen3 <think> tags
        system_prompt = card.system_prompt
        if not system_prompt.startswith("/no_think"):
            system_prompt = "/no_think\n" + system_prompt
        session_name = name or f"Chat with {card.name}"
        now = now_iso()
        with get_db() as db:
            db.execute(
                "INSERT INTO sessions (id, name, mode, character_ids, system_prompt, created_at, last_active) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, session_name, "chat", json.dumps([character]), system_prompt, now, now),
            )
            db.commit()
        return RedirectResponse(url=f"/sessions/{session_id}", status_code=303)

    @chamber.post("/sessions/{session_id}/delete")
    async def delete_session(session_id: str):
        with get_db() as db:
            db.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            db.commit()
        return RedirectResponse(url="/", status_code=303)

    @chamber.get("/sessions/{session_id}", response_class=HTMLResponse)
    async def chat_page(request: Request, session_id: str):
        if not _vllm_ready.is_set():
            return RedirectResponse(url=GATEWAY_URL)
        with get_db() as db:
            session = db.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not session:
                return RedirectResponse(url="/")
            turns = db.execute(
                "SELECT role, character_id, content, created_at "
                "FROM turns WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        char_ids = json.loads(session["character_ids"])
        character_name = char_ids[0] if char_ids else "unknown"
        try:
            card = load_character(character_name)
            display_name = card.name
        except FileNotFoundError:
            display_name = character_name.title()
        return templates.TemplateResponse(
            "chat.html",
            {
                "request": request,
                "session": dict(session),
                "turns": [dict(t) for t in turns],
                "character_name": display_name,
                "character_id": character_name,
            },
        )

    @chamber.post("/sessions/{session_id}/message")
    async def send_message(session_id: str, message: str = Form(...)):
        now = now_iso()
        with get_db() as db:
            session = db.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not session:
                return JSONResponse({"error": "Session not found"}, status_code=404)
            db.execute(
                "INSERT INTO turns (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, "user", message, now),
            )
            db.execute(
                "UPDATE sessions SET last_active = ?, turn_count = turn_count + 1 WHERE id = ?",
                (now, session_id),
            )
            db.commit()
            turns = db.execute(
                "SELECT role, content FROM turns WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()

        history = [{"role": t["role"], "content": t["content"]} for t in turns]
        system_prompt = session["system_prompt"]
        char_ids = json.loads(session["character_ids"])
        character = char_ids[0] if char_ids else "unknown"

        async def generate():
            if not _vllm_ready.is_set():
                yield {
                    "event": "token",
                    "data": "[Madison is composing his thoughts — the model is still loading. "
                    "Please wait a moment and try again.]",
                }
                yield {"event": "done", "data": ""}
                return

            full_response = []
            past_think = False  # Qwen3 emits <think>...</think> at the start
            think_buf = ""
            config = get_config()
            url = f"http://localhost:{VLLM_PORT}/v1/chat/completions"
            api_messages = [{"role": "system", "content": system_prompt}] + history
            payload = {
                "model": "madison",
                "messages": api_messages,
                "temperature": config.samplers.temperature,
                "top_p": config.samplers.top_p,
                "max_tokens": config.samplers.max_tokens,
                "stream": True,
            }

            try:
                async with httpx.AsyncClient(timeout=300) as client:
                    async with client.stream("POST", url, json=payload) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content")
                                if not content:
                                    continue

                                # ── Strip <think>...</think> at start of response ──
                                if not past_think:
                                    think_buf += content
                                    if "</think>" in think_buf:
                                        after = think_buf.split("</think>", 1)[1]
                                        past_think = True
                                        after = after.lstrip()
                                        if after:
                                            full_response.append(after)
                                            yield {"event": "token", "data": after}
                                    elif len(think_buf) > 50 and "<think>" not in think_buf:
                                        past_think = True
                                        full_response.append(think_buf)
                                        yield {"event": "token", "data": think_buf}
                                else:
                                    full_response.append(content)
                                    yield {"event": "token", "data": content}

                            except (json.JSONDecodeError, IndexError, KeyError):
                                continue

            except Exception as e:
                log.error("Inference error: %s", e)
                yield {"event": "token", "data": f"\n\n[Error: {e}]"}
            finally:
                if full_response:
                    assistant_content = "".join(full_response).strip()
                    response_time = now_iso()
                    with get_db() as db:
                        db.execute(
                            "INSERT INTO turns (session_id, role, character_id, content, created_at) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (session_id, "assistant", None, assistant_content, response_time),
                        )
                        db.execute(
                            "UPDATE sessions SET last_active = ?, turn_count = turn_count + 1 WHERE id = ?",
                            (response_time, session_id),
                        )
                        db.commit()
                    # Log to JSONL for evals
                    _append_eval_log(session_id, character, system_prompt, message, assistant_content)
            yield {"event": "done", "data": ""}

        return EventSourceResponse(generate())

    return chamber


# ── CLI ──


@app.local_entrypoint()
async def main():
    print("\nDeploy: modal deploy scripts/modal/serve_chamber.py")
    print(f"Gateway: https://seaberger--foundry-chamber-gateway-web.modal.run")
    print(f"Chamber: {CHAMBER_URL}")
    print(f"Export:  curl {CHAMBER_URL}/api/export > evals.jsonl")
