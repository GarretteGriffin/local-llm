from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException, Request, status

from config import settings
from server.auth_sessions import AuthSession, get_auth_session


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


def get_current_auth_session_optional(request: Request) -> Optional[AuthSession]:
    """Return the full auth session (user + token) if authenticated; otherwise None."""
    if not getattr(settings, "auth_enabled", False):
        return None

    session_id = request.cookies.get(_get_cookie_name())
    if not session_id:
        return None

    return get_auth_session(session_id)


def get_current_token_optional(request: Request) -> Optional[Dict[str, Any]]:
    """Return delegated OAuth token dict if authenticated; otherwise None."""
    st = get_current_auth_session_optional(request)
    if not st:
        return None
    return st.token


def require_current_token(request: Request) -> Dict[str, Any]:
    """Require a delegated OAuth token if auth is enabled.

    When auth is disabled, returns an empty dict.
    """
    if not getattr(settings, "auth_enabled", False):
        return {}

    tok = get_current_token_optional(request)
    if tok is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return tok


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
