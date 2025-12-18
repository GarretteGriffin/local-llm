"""Admin auth session store.

Separate from the user app's auth store to keep admin cookies/sessions isolated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Dict, Optional
import secrets


@dataclass
class AdminUser:
    tenant_id: str
    email: str
    name: str


@dataclass
class AdminAuthSession:
    session_id: str
    user: AdminUser
    created_at: datetime
    expires_at: datetime


_LOCK = RLock()
_SESSIONS: Dict[str, AdminAuthSession] = {}

# Conservative defaults
_TTL_SECONDS = 60 * 60 * 8  # 8h
_MAX_SESSIONS = 2000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_session(user: AdminUser) -> AdminAuthSession:
    with _LOCK:
        # Basic cap to prevent unbounded growth
        if len(_SESSIONS) >= _MAX_SESSIONS:
            # Drop oldest
            oldest = sorted(_SESSIONS.values(), key=lambda s: s.created_at)[: max(1, _MAX_SESSIONS // 10)]
            for s in oldest:
                _SESSIONS.pop(s.session_id, None)

        session_id = secrets.token_urlsafe(32)
        now = _utcnow()
        sess = AdminAuthSession(
            session_id=session_id,
            user=user,
            created_at=now,
            expires_at=now + timedelta(seconds=_TTL_SECONDS),
        )
        _SESSIONS[session_id] = sess
        return sess


def get_session(session_id: str) -> Optional[AdminAuthSession]:
    if not session_id:
        return None

    with _LOCK:
        sess = _SESSIONS.get(session_id)
        if not sess:
            return None
        if sess.expires_at <= _utcnow():
            _SESSIONS.pop(session_id, None)
            return None
        return sess


def delete_session(session_id: str) -> None:
    with _LOCK:
        _SESSIONS.pop(session_id, None)
