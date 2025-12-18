from __future__ import annotations

from dataclasses import dataclass
import secrets
import time
from typing import Any, Dict, Optional

from config import settings


@dataclass
class AuthSession:
    user: Dict[str, Any]
    token: Dict[str, Any]
    created_at: float
    last_access: float


_auth_sessions: Dict[str, AuthSession] = {}
_last_cleanup_ts: float = 0.0


def _cleanup_sessions() -> None:
    global _last_cleanup_ts

    now = time.time()
    interval = max(1, int(getattr(settings, "auth_session_cleanup_interval_seconds", 60)))
    if now - _last_cleanup_ts < interval:
        return
    _last_cleanup_ts = now

    ttl = max(60, int(getattr(settings, "auth_session_ttl_seconds", 60 * 60 * 8)))
    expired = [sid for sid, st in _auth_sessions.items() if now - st.last_access > ttl]
    for sid in expired:
        _auth_sessions.pop(sid, None)

    max_sessions = max(1, int(getattr(settings, "auth_session_max_sessions", 2000)))
    if len(_auth_sessions) <= max_sessions:
        return

    over = len(_auth_sessions) - max_sessions
    for sid, _ in sorted(_auth_sessions.items(), key=lambda kv: kv[1].last_access)[:over]:
        _auth_sessions.pop(sid, None)


def create_auth_session(*, user: Dict[str, Any], token: Dict[str, Any]) -> str:
    _cleanup_sessions()
    sid = secrets.token_urlsafe(32)
    now = time.time()
    _auth_sessions[sid] = AuthSession(user=user, token=token, created_at=now, last_access=now)
    return sid


def get_auth_session(session_id: str) -> Optional[AuthSession]:
    _cleanup_sessions()
    if not session_id:
        return None
    st = _auth_sessions.get(session_id)
    if not st:
        return None
    st.last_access = time.time()
    return st


def delete_auth_session(session_id: str) -> None:
    if not session_id:
        return
    _auth_sessions.pop(session_id, None)
