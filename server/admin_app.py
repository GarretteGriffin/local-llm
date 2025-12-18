from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse

from server.admin_settings import admin_settings
from server.routers import admin as admin_router
from server.routers import admin_auth as admin_auth_router
from server.admin_auth import get_current_admin_user_optional


app = FastAPI(
    title="Local LLM Admin",
    version="1.0.0",
    docs_url="/docs" if admin_settings.environment == "development" else None,
    redoc_url="/redoc" if admin_settings.environment == "development" else None,
)

# Session middleware is used by Authlib to store OAuth state.
if admin_settings.admin_auth_enabled:
    if not admin_settings.session_middleware_secret_key:
        raise RuntimeError(
            "ADMIN_AUTH_ENABLED=true requires SESSION_MIDDLEWARE_SECRET_KEY to be set"
        )
    app.add_middleware(
        SessionMiddleware,
        secret_key=admin_settings.session_middleware_secret_key,
        same_site="lax",
        https_only=False,
    )

_repo_root = Path(__file__).resolve().parents[1]
_admin_web_dir = _repo_root / "web_admin"
_user_web_dir = _repo_root / "web"

# Serve admin assets separately from the user UI assets.
app.mount("/static", StaticFiles(directory=str(_admin_web_dir), html=False), name="admin-static")
app.mount("/ui-static", StaticFiles(directory=str(_user_web_dir), html=False), name="ui-static")

app.include_router(admin_router.router)
app.include_router(admin_auth_router.router)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    if admin_settings.admin_auth_enabled:
        user = get_current_admin_user_optional(request)
        if not user:
            return RedirectResponse(url="/auth/login", status_code=302)

    index_path = _admin_web_dir / "index.html"
    resp = HTMLResponse(index_path.read_text(encoding="utf-8"))

    # Ensure CSRF cookie exists for admin write operations.
    if not request.cookies.get("admin_csrf"):
        import secrets

        csrf = secrets.token_urlsafe(32)
        secure = False
        if request.headers.get("x-forwarded-proto", "").lower() == "https":
            secure = True
        elif request.url.scheme == "https":
            secure = True
        resp.set_cookie(
            "admin_csrf",
            csrf,
            httponly=False,
            secure=secure,
            samesite="lax",
            path="/",
        )

    return resp
