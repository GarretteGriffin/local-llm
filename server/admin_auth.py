"""Admin auth helpers.

The admin service can optionally enforce its own Entra OIDC login.
When running behind Application Proxy with pre-auth, you may set
`ADMIN_AUTH_ENABLED=false` and rely on Entra + group assignment at the proxy.

When enabled, this uses a separate cookie/session store from the user app.
"""

from __future__ import annotations

from fastapi import Request, HTTPException

from server.admin_settings import admin_settings
from server.admin_auth_sessions import get_session


def get_current_admin_user_optional(request: Request):
    cookie_name = admin_settings.admin_auth_cookie_name
    session_id = request.cookies.get(cookie_name)
    if not session_id:
        return None

    sess = get_session(session_id)
    return sess.user if sess else None


def require_current_admin_user(request: Request):
    if not admin_settings.admin_auth_enabled:
        return None

    user = get_current_admin_user_optional(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user
