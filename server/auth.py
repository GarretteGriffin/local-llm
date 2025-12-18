from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException, Request, status

from config import settings
from server.auth_sessions import get_auth_session


def _get_cookie_name() -> str:
    return getattr(settings, "auth_cookie_name", "auth_session")


def get_current_user_optional(request: Request) -> Optional[Dict[str, Any]]:
    """Return user claims if authenticated; otherwise None.

    If auth is disabled, returns None.
    """
    if not getattr(settings, "auth_enabled", False):
        return None

    session_id = request.cookies.get(_get_cookie_name())
    if not session_id:
        return None

    st = get_auth_session(session_id)
    if not st:
        return None

    return st.user


def require_current_user(request: Request) -> Dict[str, Any]:
    """Require authentication if auth is enabled.

    When auth is disabled, returns an empty dict.
    """
    if not getattr(settings, "auth_enabled", False):
        return {}

    user = get_current_user_optional(request)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user
