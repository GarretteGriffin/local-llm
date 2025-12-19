from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


_MICROSOFT_LINK_RE = re.compile(
    r"https?://[^\s)\]>'\"}]+",
    re.IGNORECASE,
)


def extract_microsoft_urls(text: str) -> list[str]:
    """Extract likely Microsoft file/share URLs from freeform text."""
    urls = _MICROSOFT_LINK_RE.findall(text or "")
    out: list[str] = []
    for u in urls:
        lu = u.lower()
        if any(
            host in lu
            for host in (
                ".sharepoint.com/",
                ".sharepoint-df.com/",
                "1drv.ms/",
                "onedrive.live.com/",
            )
        ):
            out.append(u)
    return out


def share_id_from_url(url: str) -> str:
    """Convert a share URL into a Microsoft Graph /shares/{shareId} identifier.

    Graph expects: u!{base64url(share_url)} with padding stripped.
    """
    raw = (url or "").strip()
    if not raw:
        raise ValueError("Missing share URL")

    enc = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")
    enc = enc.rstrip("=")
    return f"u!{enc}"


@dataclass
class DriveItemRef:
    name: str
    drive_id: str
    item_id: str
    web_url: str = ""


class MicrosoftGraphTool:
    """Minimal Microsoft Graph helper for delegated, permission-aware reads."""

    def __init__(self, *, base_url: str = "https://graph.microsoft.com/v1.0") -> None:
        self.base_url = base_url.rstrip("/")

    def get_me(self, *, access_token: str, timeout_s: float = 15.0) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {access_token}"}
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.get(f"{self.base_url}/me", headers=headers)
            resp.raise_for_status()
            return resp.json()

    def resolve_share_link(
        self,
        *,
        share_url: str,
        access_token: str,
        timeout_s: float = 20.0,
    ) -> DriveItemRef:
        sid = share_id_from_url(share_url)
        headers = {"Authorization": f"Bearer {access_token}"}
        url = f"{self.base_url}/shares/{sid}/driveItem"

        with httpx.Client(timeout=timeout_s) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json() or {}

        parent = (data.get("parentReference") or {}) if isinstance(data, dict) else {}
        drive_id = (parent.get("driveId") or "").strip()
        item_id = (data.get("id") or "").strip() if isinstance(data, dict) else ""
        name = (data.get("name") or "").strip() if isinstance(data, dict) else ""
        web_url = (data.get("webUrl") or "").strip() if isinstance(data, dict) else ""

        if not drive_id or not item_id:
            raise RuntimeError("Graph did not return a resolvable drive item")

        return DriveItemRef(name=name or "driveItem", drive_id=drive_id, item_id=item_id, web_url=web_url)

    def download_drive_item_content(
        self,
        *,
        drive_id: str,
        item_id: str,
        access_token: str,
        max_bytes: int,
        timeout_s: float = 60.0,
    ) -> bytes:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/octet-stream",
        }
        url = f"{self.base_url}/drives/{drive_id}/items/{item_id}/content"

        total = 0
        chunks: list[bytes] = []

        with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
            with client.stream("GET", url, headers=headers) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_bytes():
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise RuntimeError(f"Graph download exceeded limit ({max_bytes} bytes)")
                    chunks.append(chunk)

        return b"".join(chunks)
