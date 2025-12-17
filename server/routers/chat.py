from typing import List, Optional, Tuple, Generator
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from server.dependencies import get_session
import json
import logging

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

@router.post("/stream")
async def chat_stream(
    message: str = Form(""),
    session_id: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
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
            if lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
                image_files.append((filename, content))
            else:
                doc_files.append((filename, content))
    
    def _sse(payload: dict) -> bytes:
        return f"data: {json.dumps(payload)}\n\n".encode("utf-8")

    async def event_generator() -> Generator[bytes, None, None]:
        # Send session ID first
        yield _sse({"type": "session", "session_id": sid})

        try:
            # Note: Orchestrator.process is synchronous generator currently.
            # In a true enterprise app, we'd want this to be async or run in a threadpool.
            # For now, we iterate the generator.
            for event in state.orchestrator.process(
                query=message,
                files=doc_files if doc_files else None,
                images=image_files if image_files else None,
            ):
                # All events now stream as JSON payloads
                if event.get("type") == "content":
                    event["content"] = event["content"].replace("\r", "")
                yield _sse(event)

        except Exception as e:
            logger.error(f"Error in chat stream: {e}", exc_info=True)
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
