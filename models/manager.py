"""
Model Manager - handles loading, switching, and managing multiple LLM backends.
"""
import json
import re
import asyncio
from typing import Dict, Any, List, Optional, Generator, AsyncGenerator
from pathlib import Path
from datetime import datetime

from .base import BaseBackend, ConversationHistory, Message
from .ollama_backend import OllamaBackend
from .llamacpp_backend import LlamaCppBackend
from .huggingface_backend import HuggingFaceBackend
from config import Settings, ModelConfig, GenerationParams, PresetManager

# Import web search tool
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from tools.web_search import WebSearchTool, SearchResult
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False


BACKEND_CLASSES = {
    'ollama': OllamaBackend,
    'llama_cpp': LlamaCppBackend,
    'huggingface': HuggingFaceBackend,
}


class ModelManager:
    """Manages multiple LLM backends and conversation history."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.preset_manager = PresetManager()
        
        self._backends: Dict[str, BaseBackend] = {}
        self._current_backend: Optional[BaseBackend] = None
        self._current_model_name: Optional[str] = None
        
        self._conversations: Dict[str, ConversationHistory] = {}
        self._current_conversation: Optional[ConversationHistory] = None
        
        # Web search settings
        self.web_search_enabled = False
        self.tavily_api_key = self.settings.get('tavily_api_key', None)
        self._web_search_tool: Optional[WebSearchTool] = None
        
        if WEB_SEARCH_AVAILABLE:
            self._web_search_tool = WebSearchTool(tavily_api_key=self.tavily_api_key)
        
        # Load saved conversations
        self._load_conversations()
    
    def _load_conversations(self) -> None:
        """Load saved conversations from disk."""
        conversations_dir = self.settings.conversations_directory
        for file in conversations_dir.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    conv = ConversationHistory.from_dict(data)
                    self._conversations[conv.id] = conv
            except Exception as e:
                print(f"Error loading conversation {file}: {e}")
    
    def _save_conversation(self, conversation: ConversationHistory) -> None:
        """Save a conversation to disk."""
        if not self.settings.get('save_conversations', True):
            return
        
        conversations_dir = self.settings.conversations_directory
        file_path = conversations_dir / f"{conversation.id}.json"
        
        with open(file_path, 'w') as f:
            json.dump(conversation.to_dict(), f, indent=2)
    
    def get_available_backends(self) -> List[str]:
        """Get list of available backend types."""
        return list(BACKEND_CLASSES.keys())
    
    def list_models(self, backend: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available models across backends."""
        models = []
        
        backends_to_check = [backend] if backend else self.get_available_backends()
        
        for backend_name in backends_to_check:
            if backend_name == 'ollama':
                ollama_models = OllamaBackend.list_available_models(
                    self.settings.get('ollama_host', 'http://localhost:11434')
                )
                models.extend(ollama_models)
            elif backend_name == 'llama_cpp':
                gguf_models = LlamaCppBackend.list_available_models(
                    str(self.settings.models_directory)
                )
                models.extend(gguf_models)
            elif backend_name == 'huggingface':
                hf_models = HuggingFaceBackend.list_available_models(
                    str(self.settings.models_directory)
                )
                models.extend(hf_models)
        
        # Add configured models
        for name in self.settings.list_model_configs():
            config = self.settings.get_model_config(name)
            if config:
                models.append({
                    'name': name,
                    'backend': config.backend,
                    'configured': True
                })
        
        return models
    
    def load_model(
        self,
        model_name: str,
        backend: Optional[str] = None,
        config: Optional[ModelConfig] = None
    ) -> bool:
        """Load a model by name."""
        # Try to get saved config
        if config is None:
            config = self.settings.get_model_config(model_name)
        
        if config is None:
            # Create default config
            backend = backend or self.settings.get('default_backend', 'ollama')
            config = ModelConfig(
                name=model_name,
                backend=backend,
                model_id=model_name
            )
        
        backend_type = config.backend
        
        if backend_type not in BACKEND_CLASSES:
            print(f"Unknown backend: {backend_type}")
            return False
        
        # Unload current model if different
        if self._current_backend is not None and self._current_model_name != model_name:
            self.unload_current()
        
        # Check if already loaded
        if model_name in self._backends and self._backends[model_name].is_loaded:
            self._current_backend = self._backends[model_name]
            self._current_model_name = model_name
            return True
        
        # Create and load backend
        backend_class = BACKEND_CLASSES[backend_type]
        
        if backend_type == 'ollama':
            backend_instance = backend_class(
                config,
                self.settings.get('ollama_host', 'http://localhost:11434')
            )
        else:
            backend_instance = backend_class(
                config,
                str(self.settings.models_directory)
            )
        
        if backend_instance.load():
            self._backends[model_name] = backend_instance
            self._current_backend = backend_instance
            self._current_model_name = model_name
            
            # Save config
            self.settings.set_model_config(model_name, config)
            
            return True
        
        return False
    
    def unload_current(self) -> None:
        """Unload the current model."""
        if self._current_backend is not None:
            self._current_backend.unload()
            if self._current_model_name in self._backends:
                del self._backends[self._current_model_name]
            self._current_backend = None
            self._current_model_name = None
    
    def unload_all(self) -> None:
        """Unload all models."""
        for backend in self._backends.values():
            backend.unload()
        self._backends.clear()
        self._current_backend = None
        self._current_model_name = None
    
    def get_current_model_info(self) -> Optional[Dict[str, Any]]:
        """Get info about the current model."""
        if self._current_backend is None:
            return None
        return {
            'name': self._current_model_name,
            **self._current_backend.get_info()
        }
    
    # Conversation management
    
    def new_conversation(self, system_prompt: Optional[str] = None) -> ConversationHistory:
        """Create a new conversation."""
        conv = ConversationHistory(
            max_messages=self.settings.get('max_conversation_history', 100)
        )
        
        # Set system prompt
        if system_prompt:
            conv.set_system_prompt(system_prompt)
        elif self._current_backend and self._current_backend.model_config:
            conv.set_system_prompt(self._current_backend.model_config.system_prompt)
        
        self._conversations[conv.id] = conv
        self._current_conversation = conv
        self._save_conversation(conv)
        
        return conv
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationHistory]:
        """Get a conversation by ID."""
        return self._conversations.get(conversation_id)
    
    def set_current_conversation(self, conversation_id: str) -> bool:
        """Set the current conversation."""
        if conversation_id in self._conversations:
            self._current_conversation = self._conversations[conversation_id]
            return True
        return False
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations."""
        return [
            {
                'id': conv.id,
                'title': conv.title or f"Conversation {i+1}",
                'created_at': conv.created_at.isoformat(),
                'message_count': len(conv.messages)
            }
            for i, conv in enumerate(self._conversations.values())
        ]
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        if conversation_id in self._conversations:
            # Remove from memory
            conv = self._conversations.pop(conversation_id)
            
            # Remove file
            file_path = self.settings.conversations_directory / f"{conversation_id}.json"
            if file_path.exists():
                file_path.unlink()
            
            # Clear current if needed
            if self._current_conversation and self._current_conversation.id == conversation_id:
                self._current_conversation = None
            
            return True
        return False
    
    # Web Search
    
    def enable_web_search(self, enabled: bool = True, tavily_api_key: Optional[str] = None) -> bool:
        """Enable or disable web search capability."""
        if not WEB_SEARCH_AVAILABLE:
            print("Web search tools not available. Check tools/web_search.py")
            return False
        
        self.web_search_enabled = enabled
        
        if tavily_api_key:
            self.tavily_api_key = tavily_api_key
            self.settings.set('tavily_api_key', tavily_api_key)
            self._web_search_tool = WebSearchTool(tavily_api_key=tavily_api_key)
        
        return True
    
    def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform a web search and return results."""
        if not WEB_SEARCH_AVAILABLE or not self._web_search_tool:
            return []
        
        results = self._web_search_tool.search(query, max_results)
        return [r.to_dict() for r in results]
    
    async def asearch_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform an async web search and return results."""
        if not WEB_SEARCH_AVAILABLE or not self._web_search_tool:
            return []
        
        results = await self._web_search_tool.asearch(query, max_results)
        return [r.to_dict() for r in results]
    
    def _should_search(self, message: str) -> Optional[str]:
        """Determine if a message needs web search and extract search query."""
        # Keywords that suggest web search is needed
        search_triggers = [
            r'\b(search|look up|find|google|what is|who is|when did|latest|current|recent|news|today)\b',
            r'\b(2024|2025|yesterday|this week|this month)\b',
            r'\?(.*)(price|weather|score|result|happening)\b',
        ]
        
        message_lower = message.lower()
        
        for pattern in search_triggers:
            if re.search(pattern, message_lower):
                # Extract a search query from the message
                # Remove common prefixes
                query = re.sub(r'^(please |can you |could you |)?(search for |look up |find |tell me about |what is |who is )', '', message_lower)
                query = re.sub(r'\?$', '', query).strip()
                return query if query else message
        
        return None
    
    def _format_search_context(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format search results as context for the LLM."""
        if not results:
            return ""
        
        context = f"\n\n---\n📡 **Web Search Results** (Query: \"{query}\")\n"
        context += f"*Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
        
        for i, result in enumerate(results[:5], 1):
            context += f"**[{i}] {result.get('title', 'Untitled')}**\n"
            if result.get('url'):
                context += f"Source: {result['url']}\n"
            context += f"{result.get('snippet', '')}\n\n"
        
        context += "---\n\n*Use the search results above to provide accurate, up-to-date information. Cite sources when appropriate.*\n"
        
        return context
    
    # Generation
    
    def chat(
        self,
        message: str,
        conversation: Optional[ConversationHistory] = None,
        stream: bool = True,
        use_web_search: Optional[bool] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Send a message and get a streaming response."""
        if self._current_backend is None:
            yield "Error: No model loaded. Please load a model first."
            return
        
        # Use provided conversation or current
        conv = conversation or self._current_conversation
        if conv is None:
            conv = self.new_conversation()
        
        # Check if we should perform web search
        search_context = ""
        should_search = use_web_search if use_web_search is not None else self.web_search_enabled
        
        if should_search and WEB_SEARCH_AVAILABLE and self._web_search_tool:
            search_query = self._should_search(message)
            if search_query or use_web_search:
                query = search_query or message
                yield f"🔍 Searching the web for: {query}...\n\n"
                
                try:
                    results = self.search_web(query, max_results=5)
                    if results:
                        search_context = self._format_search_context(results, query)
                        yield f"📡 Found {len(results)} results. Processing...\n\n"
                except Exception as e:
                    yield f"⚠️ Search failed: {e}\n\n"
        
        # Add user message (include search context if available)
        if search_context:
            enhanced_message = f"{message}\n{search_context}"
            conv.add_message('user', enhanced_message, original_message=message, had_web_search=True)
        else:
            conv.add_message('user', message)
        
        # Get messages for API
        messages = conv.get_messages_for_api()
        
        # Generate response
        full_response = ""
        try:
            if stream:
                for chunk in self._current_backend.generate_stream(messages, **kwargs):
                    full_response += chunk
                    yield chunk
            else:
                full_response = self._current_backend.generate(messages, **kwargs)
                yield full_response
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            yield error_msg
            full_response = error_msg
        
        # Add assistant response
        conv.add_message('assistant', full_response)
        self._save_conversation(conv)
    
    async def achat(
        self,
        message: str,
        conversation: Optional[ConversationHistory] = None,
        stream: bool = True,
        use_web_search: Optional[bool] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Send a message and get an async streaming response."""
        if self._current_backend is None:
            yield "Error: No model loaded. Please load a model first."
            return
        
        conv = conversation or self._current_conversation
        if conv is None:
            conv = self.new_conversation()
        
        # Check if we should perform web search
        search_context = ""
        should_search = use_web_search if use_web_search is not None else self.web_search_enabled
        
        if should_search and WEB_SEARCH_AVAILABLE and self._web_search_tool:
            search_query = self._should_search(message)
            if search_query or use_web_search:
                query = search_query or message
                yield f"🔍 Searching the web for: {query}...\n\n"
                
                try:
                    results = await self.asearch_web(query, max_results=5)
                    if results:
                        search_context = self._format_search_context(results, query)
                        yield f"📡 Found {len(results)} results. Processing...\n\n"
                except Exception as e:
                    yield f"⚠️ Search failed: {e}\n\n"
        
        # Add user message (include search context if available)
        if search_context:
            enhanced_message = f"{message}\n{search_context}"
            conv.add_message('user', enhanced_message, original_message=message, had_web_search=True)
        else:
            conv.add_message('user', message)
        
        messages = conv.get_messages_for_api()
        
        full_response = ""
        try:
            if stream:
                async for chunk in self._current_backend.agenerate_stream(messages, **kwargs):
                    full_response += chunk
                    yield chunk
            else:
                full_response = await self._current_backend.agenerate(messages, **kwargs)
                yield full_response
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            yield error_msg
            full_response = error_msg
        
        conv.add_message('assistant', full_response)
        self._save_conversation(conv)
    
    def apply_preset(self, preset_name: str) -> bool:
        """Apply a preset to the current model config."""
        preset = self.preset_manager.get(preset_name)
        if preset is None:
            return False
        
        if self._current_backend and self._current_backend.model_config:
            config = self._current_backend.model_config
            config.generation_params = preset.generation_params
            if preset.system_prompt:
                config.system_prompt = preset.system_prompt
            
            # Update current conversation's system prompt
            if self._current_conversation and preset.system_prompt:
                self._current_conversation.set_system_prompt(preset.system_prompt)
            
            self.settings.set_model_config(self._current_model_name, config)
            return True
        
        return False
    
    def update_generation_params(self, **kwargs) -> bool:
        """Update generation parameters for the current model."""
        if self._current_backend and self._current_backend.model_config:
            params = self._current_backend.model_config.generation_params
            for key, value in kwargs.items():
                if hasattr(params, key):
                    setattr(params, key, value)
            
            self.settings.set_model_config(
                self._current_model_name,
                self._current_backend.model_config
            )
            return True
        return False
