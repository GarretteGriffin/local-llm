"""
AI Assistant - Intelligent Local LLM Application

A ChatGPT-like experience running entirely on your machine.
- Automatic model selection based on query complexity
- Automatic web search for current information
- Document reading (Office, PDF, databases, QVD)
- Image understanding
- All settings controlled from backend - users see clean interface
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from ui import launch_ui


def check_ollama():
    """Check if Ollama is running"""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        return response.status_code == 200
    except:
        return False


def check_models():
    """Check which required models are installed"""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [m["name"] for m in models]
    except:
        pass
    return []


def print_banner():
    """Print startup banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   🤖 AI Assistant - Intelligent Local LLM                     ║
    ║                                                               ║
    ║   Features:                                                   ║
    ║   • Automatic model selection (quick/standard/power)          ║
    ║   • Automatic web search for current information              ║
    ║   • Document reading (Office, PDF, CSV, databases)            ║
    ║   • QVD file support (QlikView data)                          ║
    ║   • Image understanding and analysis                          ║
    ║                                                               ║
    ║   All intelligence is backend-controlled.                     ║
    ║   Users see a simple, clean chat interface.                   ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_model_status(installed_models):
    """Print model installation status"""
    print("\n📊 Model Configuration:")
    print("-" * 50)
    
    for tier, config in settings.models.items():
        name = config.name
        base_name = name.split(":")[0]
        
        # Check if installed
        is_installed = (
            name in installed_models or 
            f"{base_name}:latest" in installed_models or
            any(base_name in m for m in installed_models)
        )
        
        status = "✅ Installed" if is_installed else "❌ Not installed"
        print(f"   {tier.value.upper():10} → {name:20} {status}")
    
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
        print("\n💡 To install missing models, run:")
        for tier, name in missing:
            print(f"   ollama pull {name}")
        print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI Assistant - Intelligent Local LLM")
    parser.add_argument("--port", type=int, default=7860, help="UI port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create public URL")
    parser.add_argument("--check", action="store_true", help="Check model status and exit")
    args = parser.parse_args()
    
    print_banner()
    
    # Check Ollama
    if not check_ollama():
        print("❌ Ollama is not running!")
        print("   Please start Ollama first: https://ollama.ai")
        sys.exit(1)
    
    print("✅ Ollama is running")
    
    # Check models
    installed = check_models()
    print(f"✅ Found {len(installed)} installed models")
    
    print_model_status(installed)
    suggest_model_installation(installed)
    
    if args.check:
        sys.exit(0)
    
    # Launch UI
    print(f"\n🚀 Starting AI Assistant on http://localhost:{args.port}")
    print("   Press Ctrl+C to stop\n")
    
    launch_ui(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
