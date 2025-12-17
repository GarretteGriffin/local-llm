from __future__ import annotations

from typing import Dict
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from server.routers import chat

app = FastAPI(
    title="Local LLM Assistant",
    version="1.0.0",
    description="Enterprise-grade Local LLM Assistant",
    docs_url="/docs" if settings.environment == "development" else None,
    redoc_url="/redoc" if settings.environment == "development" else None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="web", html=False), name="static")

# Include routers
app.include_router(chat.router)

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    with open("web/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "environment": settings.environment}
