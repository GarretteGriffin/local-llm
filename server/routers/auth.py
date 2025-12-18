from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse

from config import settings
from server.auth_sessions import create_auth_session, delete_auth_session


router = APIRouter(prefix="/auth", tags=["auth"])


def _oidc_tenant() -> str:
    # Use explicit tenant ID for enterprise; allow override to common for multi-tenant if desired.
    return (settings.azure_tenant_id or "common").strip()


def _server_metadata_url() -> str:
    tenant = _oidc_tenant()
    return f"https://login.microsoftonline.com/{tenant}/v2.0/.well-known/openid-configuration"


def _redirect_uri(request: Request) -> str:
    # In production behind a proxy, prefer setting AZURE_REDIRECT_URI explicitly.
    if settings.azure_redirect_uri:
        return settings.azure_redirect_uri
    return str(request.url_for("auth_callback"))


def _secure_cookie(request: Request) -> bool:
    # If explicitly configured, honor it.
    if settings.auth_cookie_secure is not None:
        return bool(settings.auth_cookie_secure)

    # Otherwise infer from scheme/forwarded proto.
    proto = request.headers.get("x-forwarded-proto")
    if proto:
        return proto.lower() == "https"
    return request.url.scheme == "https"


def _principal_email(claims: Dict[str, Any]) -> str:
    # Common Microsoft claim fields.
    return (
        (claims.get("preferred_username") or "")
        or (claims.get("email") or "")
        or (claims.get("upn") or "")
    ).strip().lower()


def _is_allowed_user(claims: Dict[str, Any]) -> bool:
    allowed_tenants = [t.strip() for t in (settings.auth_allowed_tenant_ids or []) if t.strip()]
    if allowed_tenants:
        tid = (claims.get("tid") or "").strip()
        if not tid or tid not in allowed_tenants:
            return False

    allowed_emails = [e.strip().lower() for e in (settings.auth_allowed_emails or []) if e.strip()]
    if allowed_emails:
        email = _principal_email(claims)
        if not email or email not in allowed_emails:
            return False

    return True


def _require_auth_enabled() -> None:
    if not settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Authentication is not enabled")

    missing = []
    if not settings.azure_client_id:
        missing.append("AZURE_CLIENT_ID")
    if not settings.azure_client_secret:
        missing.append("AZURE_CLIENT_SECRET")
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing auth configuration: {', '.join(missing)}")


async def _get_oauth():
    try:
        from authlib.integrations.starlette_client import OAuth
    except Exception as e:
        raise HTTPException(status_code=500, detail="Authlib is not installed") from e

    oauth = OAuth()
    oauth.register(
        name="azure",
        client_id=settings.azure_client_id,
        client_secret=settings.azure_client_secret,
        server_metadata_url=_server_metadata_url(),
        client_kwargs={
            "scope": "openid profile email",
        },
    )
    return oauth


@router.get("/login")
async def login(request: Request):
    _require_auth_enabled()

    oauth = await _get_oauth()
    return await oauth.azure.authorize_redirect(request, _redirect_uri(request))


@router.get("/callback", name="auth_callback")
async def callback(request: Request):
    _require_auth_enabled()

    oauth = await _get_oauth()
    try:
        token = await oauth.azure.authorize_access_token(request)
        claims = await oauth.azure.parse_id_token(request, token)
    except Exception as e:
        # OAuthError is thrown by Authlib; keep error generic.
        raise HTTPException(status_code=401, detail="Authentication failed") from e

    claims_dict: Dict[str, Any] = dict(claims)
    if not _is_allowed_user(claims_dict):
        raise HTTPException(status_code=403, detail="User is not allowed")

    auth_sid = create_auth_session(user=claims_dict, token=dict(token))

    resp = RedirectResponse(url="/", status_code=302)
    resp.set_cookie(
        key=settings.auth_cookie_name,
        value=auth_sid,
        httponly=True,
        secure=_secure_cookie(request),
        samesite="lax",
        path="/",
        max_age=int(settings.auth_session_ttl_seconds),
    )
    return resp


@router.post("/logout")
async def logout(request: Request):
    # Always allow logout calls; if auth is disabled, it's a no-op.
    sid = request.cookies.get(settings.auth_cookie_name) if settings.auth_cookie_name else None
    if sid:
        delete_auth_session(sid)

    resp = JSONResponse({"ok": True})
    resp.delete_cookie(key=settings.auth_cookie_name, path="/")
    return resp


@router.get("/me")
async def me(request: Request):
    if not settings.auth_enabled:
        return {"auth_enabled": False}

    from server.auth import get_current_user_optional

    user = get_current_user_optional(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Return minimal identity info
    return {
        "auth_enabled": True,
        "user": {
            "name": user.get("name"),
            "preferred_username": user.get("preferred_username"),
            "tid": user.get("tid"),
            "oid": user.get("oid"),
        },
    }
