"""Admin Entra ID (OIDC) auth routes.

This is intentionally separate from the user app auth router so that:
- admin can run on a separate port/app
- cookies do not collide
- you can publish admin with App Proxy passthrough (app does auth)
  while publishing user app with App Proxy pre-auth (app does not auth)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import APIRouter, Request
from starlette.responses import RedirectResponse, JSONResponse

from server.admin_settings import admin_settings
from server.admin_auth_sessions import AdminUser, create_session, delete_session


router = APIRouter(prefix="/auth", tags=["admin-auth"])


def _effective_cookie_secure(request: Request) -> bool:
    if admin_settings.admin_auth_cookie_secure is not None:
        return bool(admin_settings.admin_auth_cookie_secure)

    # Best-effort: respect reverse proxies
    if request.headers.get("x-forwarded-proto", "").lower() == "https":
        return True
    if request.url.scheme == "https":
        return True
    return False


def _is_allowed_user(tenant_id: str, email: str) -> bool:
    if admin_settings.admin_allowed_tenant_ids:
        if tenant_id not in set(admin_settings.admin_allowed_tenant_ids):
            return False
    if admin_settings.admin_allowed_emails:
        if email.lower() not in {e.lower() for e in admin_settings.admin_allowed_emails}:
            return False
    return True


def _build_oauth() -> OAuth:
    if not admin_settings.azure_tenant_id:
        raise RuntimeError("ADMIN_AUTH_ENABLED=true requires AZURE_TENANT_ID")
    if not admin_settings.azure_client_id:
        raise RuntimeError("ADMIN_AUTH_ENABLED=true requires AZURE_CLIENT_ID")
    if not admin_settings.azure_client_secret:
        raise RuntimeError("ADMIN_AUTH_ENABLED=true requires AZURE_CLIENT_SECRET")

    tenant = admin_settings.azure_tenant_id
    authority = f"https://login.microsoftonline.com/{tenant}"

    oauth = OAuth()
    oauth.register(
        name="entra",
        client_id=admin_settings.azure_client_id,
        client_secret=admin_settings.azure_client_secret,
        server_metadata_url=f"{authority}/v2.0/.well-known/openid-configuration",
        client_kwargs={"scope": "openid profile email"},
    )
    return oauth


@router.get("/login")
async def login(request: Request):
    if not admin_settings.admin_auth_enabled:
        return JSONResponse({"detail": "Admin auth is disabled"}, status_code=400)

    oauth = _build_oauth()
    redirect_uri = admin_settings.azure_redirect_uri or str(request.url_for("callback"))
    return await oauth.entra.authorize_redirect(request, redirect_uri)


@router.get("/callback", name="callback")
async def callback(request: Request):
    if not admin_settings.admin_auth_enabled:
        return JSONResponse({"detail": "Admin auth is disabled"}, status_code=400)

    oauth = _build_oauth()

    try:
        token = await oauth.entra.authorize_access_token(request)
    except OAuthError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Authlib normalizes ID token claims into userinfo
    userinfo: Dict[str, Any] = token.get("userinfo") or {}

    tenant_id = (
        userinfo.get("tid")
        or (token.get("id_token_claims") or {}).get("tid")
        or ""
    )
    email = userinfo.get("preferred_username") or userinfo.get("email") or ""
    name = userinfo.get("name") or email

    if not tenant_id or not email:
        return JSONResponse({"error": "Missing tenant/user info in token"}, status_code=400)

    if not _is_allowed_user(tenant_id, email):
        return JSONResponse({"error": "Not authorized"}, status_code=403)

    sess = create_session(AdminUser(tenant_id=tenant_id, email=email, name=name))

    resp = RedirectResponse(url="/", status_code=302)
    resp.set_cookie(
        admin_settings.admin_auth_cookie_name,
        sess.session_id,
        httponly=True,
        secure=_effective_cookie_secure(request),
        samesite="lax",
        path="/",
    )

    # CSRF token cookie for admin write operations
    import secrets
    csrf = secrets.token_urlsafe(32)
    resp.set_cookie(
        "admin_csrf",
        csrf,
        httponly=False,
        secure=_effective_cookie_secure(request),
        samesite="lax",
        path="/",
    )

    return resp


@router.post("/logout")
async def logout(request: Request):
    cookie_name = admin_settings.admin_auth_cookie_name
    session_id: Optional[str] = request.cookies.get(cookie_name)
    if session_id:
        delete_session(session_id)

    resp = JSONResponse({"ok": True})
    resp.delete_cookie(cookie_name, path="/")
    resp.delete_cookie("admin_csrf", path="/")
    return resp


@router.get("/me")
async def me(request: Request):
    from server.admin_auth import get_current_admin_user_optional

    user = get_current_admin_user_optional(request)
    if not user:
        return JSONResponse({"authenticated": False})
    return JSONResponse(
        {
            "authenticated": True,
            "tenant_id": user.tenant_id,
            "email": user.email,
            "name": user.name,
        }
    )
