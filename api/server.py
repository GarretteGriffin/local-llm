"""
FastAPI server for the Local LLM application.
Provides REST API endpoints for model management and chat.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ModelManager
from config import Settings, ModelConfig, GenerationParams


# Pydantic models for API

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID to continue")
    stream: bool = Field(True, description="Whether to stream the response")
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1, le=32768)
    top_p: Optional[float] = Field(None, ge=0, le=1)
    top_k: Optional[int] = Field(None, ge=1)
    system_prompt: Optional[str] = Field(None, description="Override system prompt")
    use_web_search: Optional[bool] = Field(None, description="Enable web search for this request")


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    model: str


class ModelLoadRequest(BaseModel):
    model_name: str = Field(..., description="Name/ID of the model to load")
    backend: Optional[str] = Field(None, description="Backend to use: 'ollama', 'llama_cpp', 'huggingface'")
    model_path: Optional[str] = Field(None, description="Path to local model file")
    context_length: int = Field(4096, description="Context window size")
    gpu_layers: int = Field(-1, description="Number of GPU layers (-1 for auto)")
    system_prompt: str = Field("You are a helpful AI assistant.", description="System prompt")


class GenerationParamsUpdate(BaseModel):
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1, le=32768)
    top_p: Optional[float] = Field(None, ge=0, le=1)
    top_k: Optional[int] = Field(None, ge=1)
    repeat_penalty: Optional[float] = Field(None, ge=1, le=2)


class ConversationCreate(BaseModel):
    system_prompt: Optional[str] = None
    title: Optional[str] = None


class WebSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, ge=1, le=10, description="Maximum number of results")


class WebSearchSettings(BaseModel):
    enabled: bool = Field(True, description="Enable or disable web search")
    tavily_api_key: Optional[str] = Field(None, description="Tavily API key for better results")


# Global manager instance
_manager: Optional[ModelManager] = None


def get_manager() -> ModelManager:
    """Get or create the model manager."""
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Local LLM API",
        description="A fully customizable, multi-model local LLM server",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Health check
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Check if the API is running."""
        manager = get_manager()
        return {
            "status": "healthy",
            "model_loaded": manager.get_current_model_info() is not None
        }
    
    # Model endpoints
    
    @app.get("/models", tags=["Models"])
    async def list_models(backend: Optional[str] = None):
        """List all available models."""
        manager = get_manager()
        return {"models": manager.list_models(backend)}
    
    @app.get("/models/current", tags=["Models"])
    async def get_current_model():
        """Get information about the currently loaded model."""
        manager = get_manager()
        info = manager.get_current_model_info()
        if info is None:
            raise HTTPException(status_code=404, detail="No model loaded")
        return info
    
    @app.post("/models/load", tags=["Models"])
    async def load_model(request: ModelLoadRequest):
        """Load a model."""
        manager = get_manager()
        
        config = ModelConfig(
            name=request.model_name,
            backend=request.backend or manager.settings.get('default_backend', 'ollama'),
            model_path=request.model_path,
            model_id=request.model_name,
            context_length=request.context_length,
            gpu_layers=request.gpu_layers,
            system_prompt=request.system_prompt
        )
        
        success = manager.load_model(request.model_name, config=config)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load model")
        
        return {"message": f"Model '{request.model_name}' loaded successfully"}
    
    @app.post("/models/unload", tags=["Models"])
    async def unload_model():
        """Unload the current model."""
        manager = get_manager()
        manager.unload_current()
        return {"message": "Model unloaded"}
    
    @app.get("/models/backends", tags=["Models"])
    async def list_backends():
        """List available backends."""
        manager = get_manager()
        return {"backends": manager.get_available_backends()}
    
    # Web Search endpoints
    
    @app.post("/search", tags=["Web Search"])
    async def search_web(request: WebSearchRequest):
        """Search the web and return results."""
        manager = get_manager()
        results = await manager.asearch_web(request.query, request.max_results)
        return {"query": request.query, "results": results}
    
    @app.post("/search/settings", tags=["Web Search"])
    async def update_search_settings(settings: WebSearchSettings):
        """Update web search settings."""
        manager = get_manager()
        success = manager.enable_web_search(settings.enabled, settings.tavily_api_key)
        return {
            "message": "Search settings updated",
            "enabled": settings.enabled,
            "has_tavily_key": bool(settings.tavily_api_key)
        }
    
    @app.get("/search/settings", tags=["Web Search"])
    async def get_search_settings():
        """Get current web search settings."""
        manager = get_manager()
        return {
            "enabled": manager.web_search_enabled,
            "has_tavily_key": bool(manager.tavily_api_key)
        }
    
    # Generation parameters
    
    @app.get("/params", tags=["Parameters"])
    async def get_generation_params():
        """Get current generation parameters."""
        manager = get_manager()
        info = manager.get_current_model_info()
        if info is None:
            raise HTTPException(status_code=404, detail="No model loaded")
        
        config = manager._current_backend.model_config
        return config.generation_params.to_dict()
    
    @app.patch("/params", tags=["Parameters"])
    async def update_generation_params(params: GenerationParamsUpdate):
        """Update generation parameters."""
        manager = get_manager()
        
        updates = {k: v for k, v in params.dict().items() if v is not None}
        if not updates:
            raise HTTPException(status_code=400, detail="No parameters provided")
        
        success = manager.update_generation_params(**updates)
        if not success:
            raise HTTPException(status_code=404, detail="No model loaded")
        
        return {"message": "Parameters updated", "updated": updates}
    
    # Presets
    
    @app.get("/presets", tags=["Presets"])
    async def list_presets():
        """List all available presets."""
        manager = get_manager()
        presets = manager.preset_manager.get_all()
        return {
            "presets": [
                {
                    "name": p.name,
                    "description": p.description,
                    "is_builtin": manager.preset_manager.is_builtin(p.name)
                }
                for p in presets.values()
            ]
        }
    
    @app.post("/presets/{preset_name}/apply", tags=["Presets"])
    async def apply_preset(preset_name: str):
        """Apply a preset to the current model."""
        manager = get_manager()
        
        if not manager.apply_preset(preset_name):
            raise HTTPException(status_code=404, detail="Preset not found or no model loaded")
        
        return {"message": f"Preset '{preset_name}' applied"}
    
    # Chat endpoints
    
    @app.post("/chat", tags=["Chat"])
    async def chat(request: ChatRequest):
        """Send a chat message and get a response."""
        manager = get_manager()
        
        if manager.get_current_model_info() is None:
            raise HTTPException(status_code=400, detail="No model loaded")
        
        # Set conversation
        if request.conversation_id:
            if not manager.set_current_conversation(request.conversation_id):
                raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Override system prompt if provided
        if request.system_prompt and manager._current_conversation:
            manager._current_conversation.set_system_prompt(request.system_prompt)
        
        # Build kwargs
        kwargs = {}
        if request.temperature is not None:
            kwargs['temperature'] = request.temperature
        if request.max_tokens is not None:
            kwargs['max_tokens'] = request.max_tokens
        if request.top_p is not None:
            kwargs['top_p'] = request.top_p
        if request.top_k is not None:
            kwargs['top_k'] = request.top_k
        if request.use_web_search is not None:
            kwargs['use_web_search'] = request.use_web_search
        
        if request.stream:
            # Streaming response
            async def generate():
                async for chunk in manager.achat(request.message, **kwargs):
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                yield f"data: {json.dumps({'done': True, 'conversation_id': manager._current_conversation.id})}\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            response_text = ""
            async for chunk in manager.achat(request.message, stream=False, **kwargs):
                response_text += chunk
            
            return ChatResponse(
                response=response_text,
                conversation_id=manager._current_conversation.id,
                model=manager._current_model_name
            )
    
    @app.post("/chat/complete", tags=["Chat"])
    async def chat_complete(messages: List[ChatMessage], **kwargs):
        """OpenAI-compatible chat completion endpoint."""
        manager = get_manager()
        
        if manager.get_current_model_info() is None:
            raise HTTPException(status_code=400, detail="No model loaded")
        
        # Convert to format expected by backend
        message_dicts = [{"role": m.role, "content": m.content} for m in messages]
        
        # Generate response
        response = manager._current_backend.generate(message_dicts)
        
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "model": manager._current_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
                }
            ]
        }
    
    # Conversation endpoints
    
    @app.get("/conversations", tags=["Conversations"])
    async def list_conversations():
        """List all conversations."""
        manager = get_manager()
        return {"conversations": manager.list_conversations()}
    
    @app.post("/conversations", tags=["Conversations"])
    async def create_conversation(request: ConversationCreate):
        """Create a new conversation."""
        manager = get_manager()
        conv = manager.new_conversation(request.system_prompt)
        if request.title:
            conv.title = request.title
        return {"conversation_id": conv.id, "title": conv.title}
    
    @app.get("/conversations/{conversation_id}", tags=["Conversations"])
    async def get_conversation(conversation_id: str):
        """Get a conversation by ID."""
        manager = get_manager()
        conv = manager.get_conversation(conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv.to_dict()
    
    @app.delete("/conversations/{conversation_id}", tags=["Conversations"])
    async def delete_conversation(conversation_id: str):
        """Delete a conversation."""
        manager = get_manager()
        if not manager.delete_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"message": "Conversation deleted"}
    
    @app.post("/conversations/{conversation_id}/clear", tags=["Conversations"])
    async def clear_conversation(conversation_id: str):
        """Clear a conversation's messages (keep system prompt)."""
        manager = get_manager()
        conv = manager.get_conversation(conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conv.clear()
        return {"message": "Conversation cleared"}
    
    return app


def run_api_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the API server."""
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_api_server()
