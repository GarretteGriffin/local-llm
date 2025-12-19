"""Admin-managed runtime overrides.

Enterprise intent:
- Overrides are edited via the admin service and stored on disk.
- The user-facing chat service reads overrides at runtime (no restart required).
- Writes are atomic and audited.

This module is intentionally dependency-light and safe to import from both
user and admin apps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional
import logging
import json


_OVERRIDES_VERSION = 1

# Stored alongside the config package for simple ops and backups.
_CONFIG_DIR = Path(__file__).resolve().parent
_OVERRIDES_PATH = _CONFIG_DIR / "admin_overrides.json"
_AUDIT_LOG_PATH = _CONFIG_DIR / "admin_overrides_audit.log"

_LOCK = RLock()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OverridesSnapshot:
    """In-memory snapshot of overrides."""

    version: int
    data: Dict[str, Any]
    loaded_at_utc: datetime
    source_mtime_ns: Optional[int]


_CACHE: Optional[OverridesSnapshot] = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        return {}
    return data


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _append_audit(entry: Dict[str, Any]) -> None:
    try:
        line = json.dumps(entry, ensure_ascii=False)
        _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Audit should never break production traffic.
        logger.exception("Failed to append admin overrides audit entry")


def load_overrides(force_reload: bool = False) -> OverridesSnapshot:
    """Load overrides from disk with a lightweight mtime cache."""
    global _CACHE

    with _LOCK:
        try:
            mtime_ns = _OVERRIDES_PATH.stat().st_mtime_ns if _OVERRIDES_PATH.exists() else None
        except Exception:
            mtime_ns = None

        if not force_reload and _CACHE and _CACHE.source_mtime_ns == mtime_ns:
            return _CACHE

        data = _read_json_file(_OVERRIDES_PATH)
        version = int(data.get("version", _OVERRIDES_VERSION)) if isinstance(data.get("version", _OVERRIDES_VERSION), int) else _OVERRIDES_VERSION

        # Ensure stable structure.
        if "system_prompts" in data and not isinstance(data["system_prompts"], dict):
            data.pop("system_prompts", None)
        if "models" in data and not isinstance(data["models"], dict):
            data.pop("models", None)
        if "tools" in data and not isinstance(data["tools"], dict):
            data.pop("tools", None)

        snapshot = OverridesSnapshot(
            version=version,
            data=data,
            loaded_at_utc=_utcnow(),
            source_mtime_ns=mtime_ns,
        )
        _CACHE = snapshot
        return snapshot


def save_overrides(new_data: Dict[str, Any], *, updated_by: Optional[str] = None) -> OverridesSnapshot:
    """Persist overrides to disk atomically and record an audit entry."""
    if not isinstance(new_data, dict):
        raise ValueError("Overrides must be a JSON object")

    with _LOCK:
        old = load_overrides(force_reload=True).data

        merged = dict(new_data)
        merged["version"] = _OVERRIDES_VERSION
        merged["updated_at_utc"] = _utcnow().isoformat()
        if updated_by:
            merged["updated_by"] = updated_by

        content = json.dumps(merged, indent=2, ensure_ascii=False, sort_keys=True)
        _OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(_OVERRIDES_PATH, content)

        _append_audit(
            {
                "ts_utc": _utcnow().isoformat(),
                "actor": updated_by,
                "event": "admin_overrides_updated",
                "path": str(_OVERRIDES_PATH),
                "keys": sorted(list(merged.keys())),
            }
        )

        return load_overrides(force_reload=True)


def get_system_prompt(*, analyst: bool, default_assistant: str, default_analyst: str) -> str:
    snap = load_overrides()
    prompts = snap.data.get("system_prompts") or {}
    if not isinstance(prompts, dict):
        prompts = {}

    key = "analyst" if analyst else "assistant"
    value = prompts.get(key)
    if isinstance(value, str) and value.strip():
        return value

    return default_analyst if analyst else default_assistant


def get_tool_override(tool_key: str) -> Optional[Any]:
    snap = load_overrides()
    tools = snap.data.get("tools") or {}
    if not isinstance(tools, dict):
        return None
    return tools.get(tool_key)


def get_model_overrides() -> Dict[str, Dict[str, Any]]:
    snap = load_overrides()
    models = snap.data.get("models") or {}
    if not isinstance(models, dict):
        return {}

    clean: Dict[str, Dict[str, Any]] = {}
    for tier, cfg in models.items():
        if not isinstance(tier, str) or not isinstance(cfg, dict):
            continue
        clean[tier] = cfg
    return clean
