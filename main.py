"""AI Assistant - Intelligent Local LLM Application.

Runs entirely local using Ollama.

Backend features:
- Automatic model selection based on query complexity
- Automatic web search for current information (supplemental)
- Document reading (Office, PDF, databases, QVD)
- Image understanding
- All settings controlled from backend
"""
import argparse
import sys
from pathlib import Path
import logging
import subprocess
import time
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def _require_dependencies() -> None:
    """Fail fast with a friendly message if the active Python is missing deps.

    This project is typically run inside the repo virtualenv. If a user runs
    `python main.py` using a system Python, they'll get import errors.
    """

    missing = []
    for module in ("pydantic", "pydantic_settings", "fastapi", "uvicorn"):
        try:
            __import__(module)
        except ModuleNotFoundError:
            missing.append(module)

    if not missing:
        return

    msg = (
        "\nMissing required Python packages: "
        + ", ".join(missing)
        + "\n\n"
        + "This usually means you're running with the wrong Python interpreter.\n"
        + "Use the repo virtualenv interpreter instead:\n\n"
        + "  .\\venv\\Scripts\\python.exe main.py --port 7860\n\n"
        + "Or install dependencies into your current interpreter:\n\n"
        + "  python -m pip install -r requirements.txt\n\n"
    )
    print(msg)
    raise SystemExit(1)


_require_dependencies()

from config import settings
from core.logging import setup_logging

logger = logging.getLogger(__name__)


def check_ollama():
    """Check if Ollama is running"""
    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            response = client.get("http://localhost:11434/api/tags")
            return response.status_code == 200
    except Exception:
        logger.exception("Failed to check Ollama status")
        return False


def check_models():
    """Check which required models are installed"""
    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            response = client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name", "") for m in models if m.get("name")]
    except Exception:
        logger.exception("Failed to fetch installed Ollama models")
    return []


def print_banner():
    """Print startup banner"""
    banner = """
    ===============================================================

      AI Assistant - Intelligent Local LLM

      Features:
        - Automatic model selection (quick/standard/power)
        - Automatic web search for current information
        - Document reading (Office, PDF, CSV, databases)
        - QVD file support (QlikView data)
        - Image understanding and analysis

      All intelligence is backend-controlled.
      Users see a simple, clean chat interface.

    ===============================================================
    """
    print(banner)


def print_model_status(installed_models):
    """Print model installation status"""
    print("\nModel Configuration:")
    print("-" * 50)

    for tier, config in settings.models.items():
        name = config.name
        base_name = name.split(":")[0]

        # Check if installed
        is_installed = (
            name in installed_models
            or f"{base_name}:latest" in installed_models
            or any(base_name in m for m in installed_models)
        )

        status = "Installed" if is_installed else "Not installed"
        print(f"   {tier.value.upper():10} -> {name:20} {status}")

    print("-" * 50)


def suggest_model_installation(installed_models):
    """Suggest models to install"""
    missing = []
    
    for tier, config in settings.models.items():
        name = config.name
        base_name = name.split(":")[0]
        
        is_installed = (
            name in installed_models or 
            f"{base_name}:latest" in installed_models or
            any(base_name in m for m in installed_models)
        )
        
        if not is_installed:
            missing.append((tier, config.name))
    
    if missing:
        print("\nTo install missing models, run:")
        for tier, name in missing:
            print(f"   ollama pull {name}")
        print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI Assistant - Intelligent Local LLM")
    parser.add_argument("--port", type=int, default=7860, help="UI port (default: 7860)")
    parser.add_argument("--admin", action="store_true", help="Start the admin service on a separate port")
    parser.add_argument("--admin-port", type=int, default=7861, help="Admin UI port (default: 7861)")
    parser.add_argument("--check", action="store_true", help="Check model status and exit")
    args = parser.parse_args()
    
    setup_logging()
    print_banner()
    
    # Check Ollama
    if not check_ollama():
        print("Ollama is not running.")
        print("   Please start Ollama first: https://ollama.ai")
        sys.exit(1)
    
    print("Ollama is running")
    
    # Check models
    installed = check_models()
    print(f"Found {len(installed)} installed models")
    
    print_model_status(installed)
    suggest_model_installation(installed)
    
    if args.check:
        sys.exit(0)
    
    # Launch UI
    print(f"\nStarting AI Assistant on http://localhost:{args.port}")
    print("   Press Ctrl+C to stop\n")

    import uvicorn
    from server.app import app

    admin_proc: subprocess.Popen | None = None
    if args.admin:
        # Start admin as a separate process to keep concerns isolated.
        # Admin auth can be enabled via ADMIN_AUTH_ENABLED=true.
        # Use current environment and override admin port.
        env = dict(os.environ)
        env["ADMIN_PORT"] = str(args.admin_port)

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "server.admin_app:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(args.admin_port),
        ]
        admin_proc = subprocess.Popen(cmd, env=env)
        # Give a short moment for startup messages to appear.
        time.sleep(0.25)
        print(f"Admin UI available on http://localhost:{args.admin_port}")

    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
    finally:
        if admin_proc and admin_proc.poll() is None:
            admin_proc.terminate()


if __name__ == "__main__":
    main()
