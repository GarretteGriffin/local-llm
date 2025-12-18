# Local LLM Assistant

A fully customizable, multi-model local LLM application with RAG (Retrieval Augmented Generation) for intelligent document analysis.

## Features

- **Multi-Model Support**: Automatic model selection based on query complexity
  - QUICK tier: llama3.2 for fast responses
  - STANDARD/POWER tier: qwen2.5:32b for document analysis
  - VISION tier: llava for image understanding

- **RAG Pipeline**: Intelligent document retrieval using ChromaDB
  - Documents are chunked and embedded locally (nomic-embed-text)
  - Only relevant chunks sent to LLM, not entire documents
  - Handles documents of any size efficiently

- **Document Support**:
  - Office: Word (.docx), Excel (.xlsx), PowerPoint (.pptx)
  - PDF files
  - Databases: SQLite, Access, QVD (QlikView)
  - Data: CSV, JSON, XML

- **Automatic Web Search**: DuckDuckGo integration for current information

- **Image Analysis**: Upload images for visual understanding

- **Backend-Only Configuration**: Users see a clean chat interface, all settings controlled by admins

## Requirements

- Windows 10/11 (or Linux/macOS with modifications)
- Python 3.10+
- Ollama installed and running
- 32GB+ RAM recommended
- NVIDIA GPU with 24GB+ VRAM recommended

## Installation

1. Clone and setup:
```bash
git clone https://github.com/YOUR_USERNAME/local-llm.git
cd local-llm
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Pull Ollama models:
```bash
ollama pull llama3.2
ollama pull qwen2.5:32b
ollama pull llava
ollama pull nomic-embed-text
```

3. Run: `python main.py`

4. Open http://localhost:7860

## Enterprise Authentication (Microsoft Entra ID)

This project supports a **proper** Microsoft Entra ID (Azure AD) OAuth2/OIDC **Authorization Code** login flow using a **client secret**.

### 1) Create an App Registration

- Azure Portal → Microsoft Entra ID → App registrations → New registration
- Set a redirect URI:
  - Local dev: `http://localhost:7860/auth/callback`
  - Production: `https://YOUR_DOMAIN/auth/callback`
- Create a **Client Secret** (Certificates & secrets)

### 2) Configure environment variables

Create a `.env` file (not committed) or set these environment variables:

- `AUTH_ENABLED=true`
- `SESSION_MIDDLEWARE_SECRET_KEY=...` (random long string; used to sign OAuth state)
- `AZURE_TENANT_ID=...`
- `AZURE_CLIENT_ID=...`
- `AZURE_CLIENT_SECRET=...`
- `AZURE_REDIRECT_URI=...` (recommended for production)

Optional allow-lists:

- `AUTH_ALLOWED_TENANT_IDS=["<tenant-guid>"]`
- `AUTH_ALLOWED_EMAILS=["user@company.com"]`

### 3) Behavior

- When enabled, `/` redirects to `/auth/login` until authenticated.
- `/chat/stream` requires authentication (401 if not logged in).

## IT Administration (Separate Admin Service)

This repo includes a **separate admin service** intended for IT administration and safe tuning of LLM behavior. It is designed to be published separately (different URL / access policy) from the end-user chat UI.

### What it can change (runtime, no restart)

The admin UI writes an overrides file on disk:

- `config/admin_overrides.json`

The user chat service reads this file at runtime to apply:

- System prompts (Assistant vs Analyst)
- Per-tier model overrides (model name + sampling/context settings)
- Global tool toggles (e.g., disable web search)

All writes are atomic and audited to:

- `config/admin_overrides_audit.log`

### Start services

User (end-user chat) UI:

- `python main.py --port 7860`

Admin (IT) UI:

- `python main.py --admin --admin-port 7861`

Or run separately:

- `python -m uvicorn server.app:app --host 0.0.0.0 --port 7860`
- `python -m uvicorn server.admin_app:app --host 0.0.0.0 --port 7861`

Admin UI URL:

- `http://localhost:7861/`

### Admin authentication (recommended)

Enable Entra OIDC for the **admin service** with:

- `ADMIN_AUTH_ENABLED=true`
- `SESSION_MIDDLEWARE_SECRET_KEY=...`
- `AZURE_TENANT_ID=...`
- `AZURE_CLIENT_ID=...`
- `AZURE_CLIENT_SECRET=...`
- `AZURE_REDIRECT_URI=...` (set to the admin service external URL `/auth/callback`)

Optional allow-lists for admin:

- `ADMIN_ALLOWED_TENANT_IDS=["<tenant-guid>"]`
- `ADMIN_ALLOWED_EMAILS=["it.admin@company.com"]`

## Exposing to the Internet (Recommended)

For an enterprise setup, host behind **HTTPS** on a managed platform (e.g., Azure App Service / Azure Container Apps) and set:

- `AUTH_ENABLED=true`
- `AZURE_REDIRECT_URI=https://YOUR_DOMAIN/auth/callback`

If running behind a reverse proxy, ensure it forwards `X-Forwarded-Proto: https`.

### Recommended publishing with Microsoft Entra Application Proxy (Option A)

Publish **two** separate applications so access policies remain clean:

- **User app** (end users): publish `http://localhost:7860/` with Entra pre-auth and assign end-user groups.
  - In this mode you typically run the user app with `AUTH_ENABLED=false`.
- **Admin app** (IT only): publish `http://localhost:7861/` and assign only IT groups.
  - You may publish it as passthrough and use `ADMIN_AUTH_ENABLED=true`, or publish with pre-auth and set `ADMIN_AUTH_ENABLED=false`.

## License

MIT License
