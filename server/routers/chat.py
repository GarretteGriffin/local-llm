from typing import List, Optional, Tuple, AsyncGenerator
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool
from server.dependencies import get_session
from server.auth import require_current_user
import json
import logging

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


def _looks_like_image_bytes(data: bytes) -> bool:
    """Best-effort file sniffing for common image types (no external deps)."""
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

@router.post("/stream")
async def chat_stream(
    message: str = Form(""),
    session_id: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    _user: dict = Depends(require_current_user),
) -> StreamingResponse:
    """
    Stream chat response.
    Accepts message, session_id, and optional files/images.
    Returns Server-Sent Events (SSE).
    """
    sid, state = get_session(session_id)
    logger.info(f"Processing chat for session {sid}")

    doc_files: List[Tuple[str, bytes]] = []
    image_files: List[Tuple[str, bytes]] = []

    if files:
        for uf in files:
            filename = uf.filename or "upload"
            content = await uf.read()
            lower = filename.lower()
            if _looks_like_image_bytes(content) or lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
                image_files.append((filename, content))
            else:
                doc_files.append((filename, content))
    
    def _sse(payload: dict) -> bytes:
        return f"data: {json.dumps(payload)}\n\n".encode("utf-8")

    async def event_generator() -> AsyncGenerator[bytes, None]:
        # Send session ID first
        yield _sse({"type": "session", "session_id": sid})

        try:
            # Orchestrator.process is a synchronous generator; iterate it in a threadpool
            # so we don't block the FastAPI event loop.
            iterator = state.orchestrator.process(
                query=message,
                files=doc_files if doc_files else None,
                images=image_files if image_files else None,
            )

            async for event in iterate_in_threadpool(iterator):
                # All events now stream as JSON payloads
                if event.get("type") == "content":
                    event["content"] = event["content"].replace("\r", "")
                yield _sse(event)

        except Exception as e:
            logger.error(f"Error in chat stream: {e}", exc_info=True)
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
