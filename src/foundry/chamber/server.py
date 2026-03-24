"""Foundry web server — FastAPI application with Chamber chat routes."""

from __future__ import annotations

import json
import logging
import uuid
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from ..characters.loader import list_characters, load_character
from ..config import get_config
from ..db import get_db, init_db, now_iso
from ..inference.client import stream_chat

log = logging.getLogger("foundry.server")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

app = FastAPI(title="The Foundry Project")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def startup():
    log.info("Foundry starting up...")
    init_db()
    log.info("Database initialized")


@app.get("/health")
async def health():
    return {"status": "ok"}


# --- Session management routes ---


@app.get("/", response_class=HTMLResponse)
async def sessions_page(request: Request):
    """Render the sessions list page."""
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
    return templates.TemplateResponse(
        "sessions.html",
        {"request": request, "sessions": sessions, "characters": characters},
    )


@app.post("/sessions")
async def create_session(character: str = Form(...), name: str = Form("")):
    """Create a new chat session with the selected character."""
    card = load_character(character)
    session_id = uuid.uuid4().hex[:12]
    session_name = name or f"Chat with {card.name}"
    now = now_iso()

    with get_db() as db:
        db.execute(
            "INSERT INTO sessions (id, name, mode, character_ids, system_prompt, created_at, last_active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                session_name,
                "chat",
                json.dumps([character]),
                card.system_prompt,
                now,
                now,
            ),
        )
        db.commit()

    return RedirectResponse(url=f"/sessions/{session_id}", status_code=303)


@app.get("/sessions/{session_id}", response_class=HTMLResponse)
async def chat_page(request: Request, session_id: str):
    """Render the chat page for a session."""
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


@app.post("/sessions/{session_id}/message")
async def send_message(session_id: str, message: str = Form(...)):
    """Accept a user message, store it, and stream the assistant response via SSE."""
    now = now_iso()

    # Load session and store user message
    with get_db() as db:
        session = db.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session:
            return {"error": "Session not found"}

        # Store user turn
        db.execute(
            "INSERT INTO turns (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "user", message, now),
        )
        db.execute(
            "UPDATE sessions SET last_active = ?, turn_count = turn_count + 1 WHERE id = ?",
            (now, session_id),
        )
        db.commit()

        # Build conversation history
        turns = db.execute(
            "SELECT role, content FROM turns WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()

    history = [{"role": t["role"], "content": t["content"]} for t in turns]
    system_prompt = session["system_prompt"]

    async def generate():
        full_response = []
        try:
            async for token in stream_chat(system_prompt, history):
                full_response.append(token)
                yield {"event": "token", "data": token}
        finally:
            # Store whatever was received, even on client disconnect
            if full_response:
                assistant_content = "".join(full_response)
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
        yield {"event": "done", "data": ""}

    return EventSourceResponse(generate())


def start_server(host: str = "0.0.0.0", port: int = 8080, open_browser: bool = True):
    """Start the Foundry server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if open_browser:
        webbrowser.open(f"http://localhost:{port}")

    uvicorn.run(app, host=host, port=port)
