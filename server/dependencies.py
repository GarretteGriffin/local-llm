from typing import Dict, Optional, Tuple
from uuid import uuid4
from dataclasses import dataclass
from fastapi import Depends, HTTPException
from core.orchestrator import Orchestrator

@dataclass
class SessionState:
    orchestrator: Orchestrator

# In-memory session storage (replace with Redis for true enterprise scale)
_sessions: Dict[str, SessionState] = {}

def get_session_store() -> Dict[str, SessionState]:
    return _sessions

def get_session(session_id: Optional[str] = None) -> Tuple[str, SessionState]:
    """
    Retrieve or create a session.
    If session_id is provided but not found, creates a new one.
    """
    store = get_session_store()
    
    if session_id and session_id in store:
        return session_id, store[session_id]
    
    # Create new session
    new_id = session_id or str(uuid4())
    # If the user provided an ID that doesn't exist, we use it (or should we generate a new one?)
    # For security, usually better to generate a new one if not found to prevent session fixation,
    # but for a local tool, using the client-provided ID is convenient for reconnection.
    # Let's stick to generating if not found to be safe, unless it was passed explicitly.
    
    state = SessionState(orchestrator=Orchestrator())
    store[new_id] = state
    return new_id, state
