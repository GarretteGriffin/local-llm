from __future__ import annotations

from pathlib import Path
from typing import Dict
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse

from config import settings
from server.routers import chat
from server.routers import auth
from server.auth import get_current_user_optional

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

# Session middleware is used by Authlib to store OAuth state (not access tokens).
if settings.auth_enabled:
    if not settings.session_middleware_secret_key:
        raise RuntimeError(
            "AUTH_ENABLED=true requires SESSION_MIDDLEWARE_SECRET_KEY to be set"
        )
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_middleware_secret_key,
        same_site="lax",
        https_only=False,  # cookies set by auth flow use their own secure flag
    )

# Mount static files
_repo_root = Path(__file__).resolve().parents[1]
_web_dir = _repo_root / "web"
app.mount("/static", StaticFiles(directory=str(_web_dir), html=False), name="static")

# Include routers
app.include_router(chat.router)
app.include_router(auth.router)

@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    if settings.auth_enabled:
        user = get_current_user_optional(request)
        if not user:
            return RedirectResponse(url="/auth/login", status_code=302)
    index_path = _web_dir / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "environment": settings.environment}
