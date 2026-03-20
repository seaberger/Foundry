"""Foundry web server — FastAPI application."""

from __future__ import annotations

import logging
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import get_config
from ..db import init_db

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
