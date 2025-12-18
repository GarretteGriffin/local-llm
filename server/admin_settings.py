"""Admin service settings.

This is intentionally separate from `config.settings.Settings` so:
- the user service can remain pre-auth behind Application Proxy
- the admin service can be separately secured and published

All values are environment-driven.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AdminSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Server
    admin_port: int = 7861
    environment: str = "development"

    # Entra ID auth for admin UI/API (recommended for enterprise)
    admin_auth_enabled: bool = False
    admin_auth_cookie_name: str = "admin_auth_session"
    admin_auth_cookie_secure: Optional[bool] = None

    azure_tenant_id: Optional[str] = None
    azure_client_id: Optional[str] = None
    azure_client_secret: Optional[str] = None
    azure_redirect_uri: Optional[str] = None

    admin_allowed_emails: List[str] = Field(default_factory=list)
    admin_allowed_tenant_ids: List[str] = Field(default_factory=list)

    # OAuth state signing (SessionMiddleware)
    session_middleware_secret_key: Optional[str] = None


admin_settings = AdminSettings()
