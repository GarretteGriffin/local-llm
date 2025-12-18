"""Admin UI + API for enterprise configuration.

This service is intentionally separate from the end-user chat UI.
It edits admin-managed overrides stored in `config/admin_overrides.json`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from config.admin_overrides import load_overrides, save_overrides
from config import settings as base_settings
from server.admin_settings import admin_settings
from server.admin_auth import require_current_admin_user


router = APIRouter(tags=["admin"])


def _csrf_ok(request: Request) -> bool:
    token_cookie = request.cookies.get("admin_csrf")
    token_header = request.headers.get("x-csrf-token")
    return bool(token_cookie and token_header and token_cookie == token_header)


@router.get("/health")
def admin_health() -> Dict[str, str]:
    return {"status": "ok", "service": "admin", "environment": admin_settings.environment}


@router.get("/api/config")
def get_config(_user=Depends(require_current_admin_user)) -> Dict[str, Any]:
    # effective config = base settings + overrides (server-side merge happens in orchestrator)
    snap = load_overrides()

    return {
        "overrides": snap.data,
        "base": {
            "models": {tier.value: cfg.model_dump(exclude={"tier"}) for tier, cfg in base_settings.models.items()},
            "web_search_enabled": base_settings.web_search_enabled,
        },
    }


@router.put("/api/config")
async def put_config(request: Request, _user=Depends(require_current_admin_user)) -> Dict[str, Any]:
    if admin_settings.admin_auth_enabled and not _user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not _csrf_ok(request):
        raise HTTPException(status_code=403, detail="CSRF check failed")

    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")

    # Normalize and validate minimal structure.
    new_data: Dict[str, Any] = {}

    if "system_prompts" in payload:
        if not isinstance(payload["system_prompts"], dict):
            raise HTTPException(status_code=400, detail="system_prompts must be an object")
        new_data["system_prompts"] = payload["system_prompts"]

    if "models" in payload:
        if not isinstance(payload["models"], dict):
            raise HTTPException(status_code=400, detail="models must be an object")
        new_data["models"] = payload["models"]

    if "tools" in payload:
        if not isinstance(payload["tools"], dict):
            raise HTTPException(status_code=400, detail="tools must be an object")
        new_data["tools"] = payload["tools"]

    actor: Optional[str] = None
    if _user is not None:
        actor = getattr(_user, "email", None)

    # Merge with existing to support partial updates.
    current = load_overrides(force_reload=True).data
    merged = dict(current)
    merged.update(new_data)

    snap = save_overrides(merged, updated_by=actor)
    return {"ok": True, "overrides": snap.data}


@router.post("/api/reset")
def reset_config(request: Request, _user=Depends(require_current_admin_user)) -> Dict[str, Any]:
    if admin_settings.admin_auth_enabled and not _user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not _csrf_ok(request):
        raise HTTPException(status_code=403, detail="CSRF check failed")

    actor: Optional[str] = None
    if _user is not None:
        actor = getattr(_user, "email", None)

    snap = save_overrides({"version": 1}, updated_by=actor)
    return {"ok": True, "overrides": snap.data}
