"""
Backend Configuration - ALL settings controlled here, not by users.
Users see a clean chat interface with zero configuration options.
"""
from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelTier(str, Enum):
    """Model tiers for intelligent routing"""
    QUICK = "quick"      # Fast responses, simple queries (weather, time, basic facts)
    STANDARD = "standard"  # General conversation, moderate complexity
    POWER = "power"      # Complex analysis, reasoning, large documents
    VISION = "vision"    # Image understanding and analysis


class ToolType(str, Enum):
    """Available tools the system can use"""
    WEB_SEARCH = "web_search"
    FILE_READER = "file_reader"
    IMAGE_VISION = "image_vision"
    DATABASE = "database"


class ModelConfig(BaseModel):
    """Configuration for a specific model"""
    name: str
    backend: str = "ollama" # "ollama", "llamacpp", "huggingface"
    tier: ModelTier
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 40
    context_window: int = 4096
    supports_vision: bool = False
    supports_streaming: bool = True


class RoutingRule(BaseModel):
    """Rules for routing queries to appropriate models"""
    keywords: List[str]
    tier: ModelTier
    tools: List[ToolType] = Field(default_factory=list)
    priority: int = 0  # Higher = checked first


class Settings(BaseSettings):
    """Master settings - all backend controlled"""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Environment
    environment: str = "development"
    log_level: str = "INFO"

    # Model configurations by tier
    models: Dict[ModelTier, ModelConfig] = Field(default_factory=dict)
    
    # Routing rules for query classification
    routing_rules: List[RoutingRule] = Field(default_factory=list)
    
    # Tool configurations
    web_search_enabled: bool = True
    web_search_provider: str = "duckduckgo"  # or "tavily"
    tavily_api_key: Optional[str] = None
    
    # File handling - MAXED for 128GB RAM
    max_file_size_mb: int = 100
    supported_extensions: List[str] = Field(default_factory=lambda: [
        # Office
        ".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt",
        # PDF
        ".pdf",
        # Data
        ".csv", ".json", ".xml",
        # Database
        ".db", ".sqlite", ".sqlite3", ".mdb", ".accdb",
        # QlikView
        ".qvd",
        # Images
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
        # Text
        ".txt", ".md", ".rtf"
    ])
    
    # Vision settings
    vision_model: Optional[str] = None
    
    # Server settings
    ui_port: int = 7860
    api_port: int = 8000
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize default models and routing rules if not set"""
        if not self.models:
            self._setup_default_models()
        if not self.routing_rules:
            self._setup_default_routing()
    
    def _setup_default_models(self):
        """Configure default models - optimized for speed and quality"""
        self.models = {
            ModelTier.QUICK: ModelConfig(
                name="qwen2.5:14b",  # UPGRADED: Llama 3.2 was too weak. 14B is fast enough (133 t/s).
                backend="ollama",
                tier=ModelTier.QUICK,
                temperature=0.7,
                max_tokens=1024,
                context_window=16384,
                top_p=0.9,
                top_k=40
            ),
            ModelTier.STANDARD: ModelConfig(
                name="qwen2.5:14b",  # Switched to 14B for better speed/quality balance
                backend="ollama",
                tier=ModelTier.STANDARD,
                temperature=0.7,  # Higher temp for better engagement
                max_tokens=2048,
                context_window=16384,
                top_p=0.9,
                top_k=40
            ),
            ModelTier.POWER: ModelConfig(
                name="qwen2.5:32b",  # Keep 32B for heavy lifting
                backend="ollama",
                tier=ModelTier.POWER,
                temperature=0.4,  # Slightly higher but still focused
                max_tokens=4096,
                context_window=32768,
                top_p=0.95,
                top_k=50
            ),
            ModelTier.VISION: ModelConfig(
                name="llava:latest",
                backend="ollama",
                tier=ModelTier.VISION,
                temperature=0.5,
                max_tokens=2048,
                context_window=4096,
                supports_vision=True,
                top_p=0.9,
                top_k=40
            )
        }
    
    def _setup_default_routing(self):
        """Configure default routing rules"""
        self.routing_rules = [
            # Greetings - NO WEB SEARCH
            RoutingRule(
                keywords=["hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye", 
                         "ok", "okay", "good morning", "good afternoon", "good evening"],
                tier=ModelTier.QUICK,
                tools=[],  # No tools needed for greetings
                priority=10
            ),
            # Simple queries - WEB SEARCH
            RoutingRule(
                keywords=["weather", "time", "date", "what is", "who is", "define", 
                         "meaning of", "translate", "convert", "calculate", "how many",
                         "what's the", "when is", "where is"],
                tier=ModelTier.QUICK,
                tools=[ToolType.WEB_SEARCH],
                priority=10
            ),
            RoutingRule(
                keywords=["analyze", "analysis", "compare", "comparison", "evaluate",
                         "fiscal", "quarterly", "revenue", "profit", "loss", "budget",
                         "forecast", "trend", "strategic", "comprehensive", "detailed",
                         "explain in depth", "breakdown", "assessment", "overhead",
                         "expenditure", "financial", "report", "summarize document",
                         "review this", "what does this mean", "implications"],
                tier=ModelTier.POWER,
                tools=[ToolType.FILE_READER, ToolType.DATABASE, ToolType.WEB_SEARCH],
                priority=20
            ),
            RoutingRule(
                keywords=["image", "picture", "photo", "screenshot", "diagram",
                         "chart", "graph", "what's in this", "describe this",
                         "read this", "ocr", "extract text from", "what do you see"],
                tier=ModelTier.VISION,
                tools=[ToolType.IMAGE_VISION],
                priority=30
            ),
            RoutingRule(
                keywords=["file", "document", "spreadsheet", "pdf", "excel", 
                         "word", "powerpoint", "database", "qvd", "open", "read"],
                tier=ModelTier.POWER,
                tools=[ToolType.FILE_READER, ToolType.DATABASE],
                priority=15
            ),
        ]
    
    def get_model_for_tier(self, tier: ModelTier) -> ModelConfig:
        """Get model configuration for a tier"""
        return self.models.get(tier, self.models[ModelTier.STANDARD])
    
    def update_model(self, tier: ModelTier, **kwargs):
        """Update model configuration for a tier"""
        if tier in self.models:
            model = self.models[tier]
            updated_data = model.model_dump()
            updated_data.update(kwargs)
            self.models[tier] = ModelConfig(**updated_data)
    
    def save(self, path: str = "config.json"):
        """Save settings to file"""
        # Pydantic models have a model_dump method
        data = self.model_dump(mode="json", exclude={"routing_rules", "supported_extensions"})
        
        export_data = {
            "models": {
                tier.value: cfg.model_dump(exclude={"tier"})
                for tier, cfg in self.models.items()
            },
            "web_search_enabled": self.web_search_enabled,
            "web_search_provider": self.web_search_provider,
            "tavily_api_key": self.tavily_api_key,
            "max_file_size_mb": self.max_file_size_mb,
            "ui_port": self.ui_port,
            "api_port": self.api_port
        }
        Path(path).write_text(json.dumps(export_data, indent=2))

    @classmethod
    def load(cls, path: str = "config.json") -> "Settings":
        """Load settings from file - helper to load from JSON on top of env vars"""
        if not Path(path).exists():
            return cls()
            
        try:
            # For legacy support, we just return default which loads from env
            return cls() 
        except Exception:
            return cls()


# Global settings instance
settings = Settings()
