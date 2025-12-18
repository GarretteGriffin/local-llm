from typing import Dict, Optional, Tuple
from uuid import uuid4
from dataclasses import dataclass
import time

from fastapi import Depends, HTTPException

from config import settings
from core.orchestrator import Orchestrator

@dataclass
class SessionState:
    orchestrator: Orchestrator
    created_at: float
    last_access: float

# In-memory session storage (replace with Redis for true enterprise scale)
_sessions: Dict[str, SessionState] = {}
_last_cleanup_ts: float = 0.0


def _cleanup_sessions(store: Dict[str, SessionState]) -> None:
    """Evict expired sessions and cap the total number of sessions."""
    global _last_cleanup_ts

    now = time.time()
    if now - _last_cleanup_ts < settings.session_cleanup_interval_seconds:
        return
    _last_cleanup_ts = now

    # TTL eviction
    ttl = max(1, int(settings.session_ttl_seconds))
    expired = [sid for sid, st in store.items() if now - st.last_access > ttl]
    for sid in expired:
        store.pop(sid, None)

    # Hard cap eviction (least-recently used)
    max_sessions = max(1, int(settings.session_max_sessions))
    if len(store) <= max_sessions:
        return

    # Remove oldest by last_access
    over = len(store) - max_sessions
    for sid, _ in sorted(store.items(), key=lambda kv: kv[1].last_access)[:over]:
        store.pop(sid, None)

def get_session_store() -> Dict[str, SessionState]:
    return _sessions

def get_session(session_id: Optional[str] = None) -> Tuple[str, SessionState]:
    """
    Retrieve or create a session.
    If session_id is provided but not found, creates a new one.
    """
    store = get_session_store()
    _cleanup_sessions(store)
    now = time.time()
    
    if session_id and session_id in store:
        state = store[session_id]
        state.last_access = now
        return session_id, state
    
    # Create new session
    # If the client provides an unknown session_id, generate a new one to avoid session fixation.
    new_id = str(uuid4())

    state = SessionState(orchestrator=Orchestrator(), created_at=now, last_access=now)
    store[new_id] = state
    return new_id, state
