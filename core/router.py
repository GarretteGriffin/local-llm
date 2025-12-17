"""
Query Router - Intelligently classifies queries and selects appropriate model + tools.
All routing logic is backend-controlled, invisible to users.
"""
import re
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass

from config import settings, ModelTier, ToolType


@dataclass
class RoutingDecision:
    """Result of query analysis"""
    tier: ModelTier
    tools: List[ToolType]
    confidence: float
    reasoning: str
    has_file: bool = False
    has_image: bool = False
    needs_web_search: bool = False


class QueryRouter:
    """
    Analyzes incoming queries and determines:
    1. Which model tier to use (quick/standard/power/vision)
    2. Which tools are needed (web search, file reader, vision, database)
    3. Whether to search the web automatically
    """
    
    # Patterns that indicate web search is needed
    WEB_SEARCH_PATTERNS = [
        r"\b(today|current|latest|recent|now|this week|this month|this year)\b",
        r"\b(weather|news|stock|price|score|result)\b",
        r"\b(what('s| is) happening|what('s| is) going on)\b",
        r"\b(who won|who is winning|who's leading)\b",
        r"\b(update|updates|status)\b",
        r"\b(2024|2025|2026)\b",  # Recent years
    ]
    
    # Patterns indicating complex analysis needed
    POWER_PATTERNS = [
        r"\b(analyze|analysis|evaluate|assessment|review)\b",
        r"\b(compare|comparison|versus|vs\.?)\b",
        r"\b(fiscal|quarter|quarterly|annual|yearly)\b",
        r"\b(financial|revenue|profit|loss|budget|overhead|expenditure)\b",
        r"\b(strategic|comprehensive|detailed|in-depth|thorough)\b",
        r"\b(summarize|summary|breakdown|explain)\b.*\b(document|file|report|data)\b",
        r"\b(implications|consequences|impact|effects)\b",
        r"\b(forecast|predict|projection|trend)\b",
    ]
    
    # Quick response patterns
    QUICK_PATTERNS = [
        r"^(hi|hello|hey|thanks|thank you|bye|goodbye|ok|okay)[\s!?.]*$",
        r"^what('s| is) (the )?(time|date)\??$",
        r"^(define|meaning of|what does .* mean)\b",
        r"^(translate|convert|calculate)\b",
        r"^how (many|much|old|tall|long|far)\b",
    ]
    
    def __init__(self):
        self.settings = settings
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency"""
        self.web_search_re = [re.compile(p, re.IGNORECASE) for p in self.WEB_SEARCH_PATTERNS]
        self.power_re = [re.compile(p, re.IGNORECASE) for p in self.POWER_PATTERNS]
        self.quick_re = [re.compile(p, re.IGNORECASE) for p in self.QUICK_PATTERNS]
    
    def route(
        self, 
        query: str, 
        has_files: bool = False,
        file_types: Optional[List[str]] = None,
        has_images: bool = False
    ) -> RoutingDecision:
        """
        Analyze query and determine routing.
        
        Args:
            query: The user's input text
            has_files: Whether files are attached
            file_types: List of file extensions attached
            has_images: Whether images are attached
            
        Returns:
            RoutingDecision with model tier, tools, and reasoning
        """
        query_lower = query.lower().strip()
        file_types = file_types or []
        
        # Determine required tools
        tools: Set[ToolType] = set()
        tier = ModelTier.STANDARD
        confidence = 0.7
        reasoning_parts = []
        
        # Check for images first - they require vision model
        if has_images or self._mentions_images(query_lower):
            tier = ModelTier.VISION
            tools.add(ToolType.IMAGE_VISION)
            confidence = 0.9
            reasoning_parts.append("Image analysis required")
        
        # Check for files
        if has_files:
            tools.add(ToolType.FILE_READER)
            reasoning_parts.append(f"File processing: {', '.join(file_types)}")
            
            # Database files need database tool
            db_extensions = {'.db', '.sqlite', '.sqlite3', '.mdb', '.accdb', '.qvd'}
            if any(ext.lower() in db_extensions for ext in file_types):
                tools.add(ToolType.DATABASE)
                reasoning_parts.append("Database access required")
        
        # Check if web search is needed (automatic, no button)
        needs_web = self._needs_web_search(query_lower)
        if needs_web and self.settings.web_search_enabled:
            tools.add(ToolType.WEB_SEARCH)
            reasoning_parts.append("Web search for current information")
        
        # Determine tier based on query complexity (if not already set to VISION)
        if tier != ModelTier.VISION:
            tier, tier_confidence, tier_reason = self._determine_tier(query_lower, has_files)
            confidence = max(confidence, tier_confidence)
            reasoning_parts.append(tier_reason)
        
        # Build final decision
        return RoutingDecision(
            tier=tier,
            tools=list(tools),
            confidence=confidence,
            reasoning=" | ".join(reasoning_parts),
            has_file=has_files,
            has_image=has_images,
            needs_web_search=needs_web
        )
    
    def _needs_web_search(self, query: str) -> bool:
        """Determine if web search is needed for this query"""
        # Check compiled patterns
        for pattern in self.web_search_re:
            if pattern.search(query):
                return True
        
        # Check routing rules
        for rule in self.settings.routing_rules:
            if ToolType.WEB_SEARCH in rule.tools:
                for keyword in rule.keywords:
                    if keyword.lower() in query:
                        return True
        
        return False
    
    def _mentions_images(self, query: str) -> bool:
        """Check if query mentions images"""
        image_words = ['image', 'picture', 'photo', 'screenshot', 'diagram', 
                      'chart', 'graph', 'what do you see', "what's in this"]
        return any(word in query for word in image_words)
    
    def _determine_tier(self, query: str, has_files: bool) -> Tuple[ModelTier, float, str]:
        """Determine which model tier to use"""
        
        # Check for quick patterns first (highest priority for simple queries)
        for pattern in self.quick_re:
            if pattern.search(query):
                return ModelTier.QUICK, 0.9, "Simple query → Quick model"
        
        # Check for power patterns
        power_matches = sum(1 for p in self.power_re if p.search(query))
        if power_matches >= 2 or has_files:
            return ModelTier.POWER, 0.85, f"Complex analysis ({power_matches} indicators) → Power model"
        
        if power_matches == 1:
            return ModelTier.STANDARD, 0.7, "Moderate complexity → Standard model"
        
        # Check routing rules from settings
        matched_rules = []
        for rule in self.settings.routing_rules:
            keyword_matches = sum(1 for kw in rule.keywords if kw.lower() in query)
            if keyword_matches > 0:
                matched_rules.append((rule, keyword_matches))
        
        if matched_rules:
            # Sort by priority and match count
            matched_rules.sort(key=lambda x: (x[0].priority, x[1]), reverse=True)
            best_rule = matched_rules[0][0]
            return best_rule.tier, 0.75, f"Matched rule keywords → {best_rule.tier.value.title()} model"
        
        # Check query length and complexity heuristics
        word_count = len(query.split())
        if word_count <= 5:
            return ModelTier.QUICK, 0.6, "Short query → Quick model"
        elif word_count >= 30:
            return ModelTier.POWER, 0.65, "Long/detailed query → Power model"
        
        # Default to standard
        return ModelTier.STANDARD, 0.5, "Default → Standard model"
    
    def explain_decision(self, decision: RoutingDecision) -> str:
        """Generate human-readable explanation of routing decision (for logging)"""
        lines = [
            f"Model Tier: {decision.tier.value.upper()}",
            f"Confidence: {decision.confidence:.0%}",
            f"Tools: {', '.join(t.value for t in decision.tools) or 'None'}",
            f"Reasoning: {decision.reasoning}"
        ]
        if decision.has_file:
            lines.append("Files attached")
        if decision.has_image:
            lines.append("Images attached")
        if decision.needs_web_search:
            lines.append("Web search enabled")
        
        return "\n".join(lines)
