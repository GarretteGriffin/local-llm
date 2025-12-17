"""
Backend Configuration - ALL settings controlled here, not by users.
Users see a clean chat interface with zero configuration options.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from pathlib import Path


class ModelTier(Enum):
    """Model tiers for intelligent routing"""
    QUICK = "quick"      # Fast responses, simple queries (weather, time, basic facts)
    STANDARD = "standard"  # General conversation, moderate complexity
    POWER = "power"      # Complex analysis, reasoning, large documents
    VISION = "vision"    # Image understanding and analysis


class ToolType(Enum):
    """Available tools the system can use"""
    WEB_SEARCH = "web_search"
    FILE_READER = "file_reader"
    IMAGE_VISION = "image_vision"
    DATABASE = "database"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    backend: str  # "ollama", "llamacpp", "huggingface"
    tier: ModelTier
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 40
    context_window: int = 4096
    supports_vision: bool = False
    supports_streaming: bool = True


@dataclass 
class RoutingRule:
    """Rules for routing queries to appropriate models"""
    keywords: List[str]
    tier: ModelTier
    tools: List[ToolType] = field(default_factory=list)
    priority: int = 0  # Higher = checked first


@dataclass
class Settings:
    """Master settings - all backend controlled"""
    
    # Model configurations by tier
    models: Dict[ModelTier, ModelConfig] = field(default_factory=dict)
    
    # Routing rules for query classification
    routing_rules: List[RoutingRule] = field(default_factory=list)
    
    # Tool configurations
    web_search_enabled: bool = True
    web_search_provider: str = "duckduckgo"  # or "tavily"
    tavily_api_key: Optional[str] = None
    
    # File handling - MAXED for 128GB RAM
    max_file_size_mb: int = 100
    supported_extensions: List[str] = field(default_factory=lambda: [
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
    
    def __post_init__(self):
        """Initialize default models and routing rules"""
        if not self.models:
            self._setup_default_models()
        if not self.routing_rules:
            self._setup_default_routing()
    
    def _setup_default_models(self):
        """Configure default models - optimized for 32B Qwen model"""
        # qwen2.5:32b - excellent for document analysis, fits well in 32GB VRAM
        # With 32B model, we can use larger context windows comfortably
        self.models = {
            ModelTier.QUICK: ModelConfig(
                name="llama3.2:latest",  # 3B - instant responses
                backend="ollama",
                tier=ModelTier.QUICK,
                temperature=0.3,
                max_tokens=1024,
                context_window=8192,
                top_p=0.9,
                top_k=40
            ),
            ModelTier.STANDARD: ModelConfig(
                name="qwen2.5:32b",  # 32B for standard queries
                backend="ollama",
                tier=ModelTier.STANDARD,
                temperature=0.5,
                max_tokens=2048,
                context_window=16384,  # 16k context
                top_p=0.9,
                top_k=40
            ),
            ModelTier.POWER: ModelConfig(
                name="qwen2.5:32b",  # 32B for document analysis
                backend="ollama",
                tier=ModelTier.POWER,
                temperature=0.2,  # Low temp for precise analysis
                max_tokens=4096,
                context_window=32768,  # 32k context - room for big docs
                top_p=0.95,
                top_k=50
            ),
            ModelTier.VISION: ModelConfig(
                name="llava:latest",
                backend="ollama",
                tier=ModelTier.VISION,
                temperature=0.4,
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
            # Quick queries - simple factual questions
            RoutingRule(
                keywords=["weather", "time", "date", "what is", "who is", "define", 
                         "meaning of", "translate", "convert", "calculate", "how many",
                         "what's the", "when is", "where is", "hello", "hi", "hey",
                         "thanks", "thank you", "bye", "goodbye"],
                tier=ModelTier.QUICK,
                tools=[ToolType.WEB_SEARCH],
                priority=10
            ),
            # Power queries - complex analysis
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
            # Vision queries - image related
            RoutingRule(
                keywords=["image", "picture", "photo", "screenshot", "diagram",
                         "chart", "graph", "what's in this", "describe this",
                         "read this", "ocr", "extract text from", "what do you see"],
                tier=ModelTier.VISION,
                tools=[ToolType.IMAGE_VISION],
                priority=30
            ),
            # File-related queries
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
            for key, value in kwargs.items():
                if hasattr(self.models[tier], key):
                    setattr(self.models[tier], key, value)
    
    def save(self, path: str = "config.json"):
        """Save settings to file"""
        data = {
            "models": {
                tier.value: {
                    "name": cfg.name,
                    "backend": cfg.backend,
                    "temperature": cfg.temperature,
                    "max_tokens": cfg.max_tokens,
                    "top_p": cfg.top_p,
                    "top_k": cfg.top_k,
                    "context_window": cfg.context_window,
                    "supports_vision": cfg.supports_vision
                }
                for tier, cfg in self.models.items()
            },
            "web_search_enabled": self.web_search_enabled,
            "web_search_provider": self.web_search_provider,
            "tavily_api_key": self.tavily_api_key,
            "max_file_size_mb": self.max_file_size_mb,
            "ui_port": self.ui_port,
            "api_port": self.api_port
        }
        Path(path).write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, path: str = "config.json") -> "Settings":
        """Load settings from file"""
        if not Path(path).exists():
            return cls()
        
        data = json.loads(Path(path).read_text())
        settings = cls()
        
        # Load models
        for tier_str, cfg in data.get("models", {}).items():
            tier = ModelTier(tier_str)
            settings.models[tier] = ModelConfig(
                name=cfg["name"],
                backend=cfg["backend"],
                tier=tier,
                temperature=cfg.get("temperature", 0.7),
                max_tokens=cfg.get("max_tokens", 2048),
                top_p=cfg.get("top_p", 0.9),
                top_k=cfg.get("top_k", 40),
                context_window=cfg.get("context_window", 4096),
                supports_vision=cfg.get("supports_vision", False)
            )
        
        # Load other settings
        settings.web_search_enabled = data.get("web_search_enabled", True)
        settings.web_search_provider = data.get("web_search_provider", "duckduckgo")
        settings.tavily_api_key = data.get("tavily_api_key")
        settings.max_file_size_mb = data.get("max_file_size_mb", 50)
        settings.ui_port = data.get("ui_port", 7860)
        settings.api_port = data.get("api_port", 8000)
        
        return settings


# Global settings instance
settings = Settings()
