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

## License

MIT License
