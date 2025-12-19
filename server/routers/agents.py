from __future__ import annotations

from typing import List, Optional, Tuple, AsyncGenerator

from fastapi import APIRouter, Depends, File, Form, UploadFile, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.concurrency import iterate_in_threadpool

from config import settings
from core.agents import list_agents
from server.dependencies import get_session
from server.auth import require_current_user, get_current_token_optional

import json
import logging

router = APIRouter(prefix="/agents", tags=["agents"])
logger = logging.getLogger(__name__)


def _looks_like_image_bytes(data: bytes) -> bool:
    if not data:
        return False
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if data.startswith(b"\xff\xd8\xff"):
        return True
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return True
    if data.startswith(b"BM"):
        return True
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return True
    return False


async def _read_upload_limited(
    uf: UploadFile,
    *,
    max_bytes: int,
    chunk_size: int,
) -> bytes:
    data = bytearray()
    total = 0
    while True:
        chunk = await uf.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload '{uf.filename or 'upload'}' exceeds limit ({max_bytes} bytes).",
            )
        data.extend(chunk)
    return bytes(data)


@router.get("")
async def agents_index():
    """List available workforce agents."""
    return {"agents": list_agents()}


@router.post("/stream")
async def agent_stream(
    request: Request,
    agent: str = Form("workforce"),
    message: str = Form(""),
    session_id: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    _user: dict = Depends(require_current_user),
) -> StreamingResponse:
    """Stream an agent-run response (SSE).

    Uses the same session store as chat to preserve document memory, but runs
    with an agent-specific tool/tier profile.
    """

    sid, state = get_session(session_id)
    token = get_current_token_optional(request) or {}

    user_oid = (_user or {}).get("oid")
    user_upn = (_user or {}).get("preferred_username") or (_user or {}).get("upn")
    logger.info(
        "Processing agent session=%s agent=%s user_oid=%s user=%s",
        sid,
        (agent or "workforce").strip().lower(),
        user_oid or "-",
        (user_upn or "-").strip().lower(),
    )

    max_total_mb = settings.max_upload_total_mb or settings.max_file_size_mb
    max_total_bytes = int(max_total_mb) * 1024 * 1024
    max_files = int(settings.max_upload_files)
    chunk_size = int(settings.upload_read_chunk_bytes)

    doc_files: List[Tuple[str, bytes]] = []
    image_files: List[Tuple[str, bytes]] = []

    total_read = 0

    if files:
        if len(files) > max_files:
            raise HTTPException(status_code=400, detail=f"Too many files uploaded (max {max_files}).")

        for uf in files:
            filename = uf.filename or "upload"
            remaining = max_total_bytes - total_read
            if remaining <= 0:
                raise HTTPException(status_code=413, detail=f"Request payload exceeds limit ({max_total_mb} MB).")

            content = await _read_upload_limited(uf, max_bytes=remaining, chunk_size=chunk_size)
            total_read += len(content)
            lower = filename.lower()
            if _looks_like_image_bytes(content) or lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
                image_files.append((filename, content))
            else:
                doc_files.append((filename, content))

    def _sse(payload: dict) -> bytes:
        return f"data: {json.dumps(payload)}\n\n".encode("utf-8")

    async def event_generator() -> AsyncGenerator[bytes, None]:
        yield _sse({"type": "session", "session_id": sid})

        try:
            iterator = state.orchestrator.process_agent(
                agent_name=agent,
                query=message,
                files=doc_files if doc_files else None,
                images=image_files if image_files else None,
                user=_user,
                token=token,
            )

            async for event in iterate_in_threadpool(iterator):
                if event.get("type") == "content" and isinstance(event.get("content"), str):
                    event["content"] = event["content"].replace("\r", "")
                yield _sse(event)

        except Exception:
            logger.exception(
                "Error in agent stream session=%s agent=%s user_oid=%s user=%s",
                sid,
                (agent or "workforce").strip().lower(),
                user_oid or "-",
                (user_upn or "-").strip().lower(),
            )
            yield _sse(
                {
                    "type": "error",
                    "message": "Something went wrong while processing that request.",
                    "where": "server",
                    "internal": True,
                }
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/disable")
async def disable_agents():
    """Placeholder endpoint for future admin control (kept minimal)."""
    return JSONResponse({"ok": True})
