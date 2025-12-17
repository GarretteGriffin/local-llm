"""Orchestrator - the brain of the system.

Coordinates routing + tools + generation.
All orchestration happens on the backend; the UI is just a client.
"""
from typing import List, Optional, Generator, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import httpx
import json

from config import settings, ModelTier, ToolType
from core.router import QueryRouter, RoutingDecision
from tools import WebSearchTool, FileReaderTool, VisionTool, DatabaseTool, RAGTool


@dataclass
class Message:
    """A chat message"""
    role: str  # "user", "assistant", "system"
    content: str
    images: Optional[List[str]] = None  # Base64 encoded images


@dataclass
class ProcessingContext:
    """Context gathered for processing a query"""
    query: str
    decision: RoutingDecision
    web_results: str = ""
    file_contents: List[str] = None
    image_analyses: List[str] = None
    rag_context: str = ""  # RAG-retrieved content
    
    def __post_init__(self):
        if self.file_contents is None:
            self.file_contents = []
        if self.image_analyses is None:
            self.image_analyses = []
    
    def build_context(self) -> str:
        """Build full context string for LLM - RAG first, then files, then web"""
        parts = []
        
        # RAG CONTEXT FIRST - most relevant chunks
        if self.rag_context:
            parts.append(self.rag_context)
        
        # Direct file contents (for small files or fallback)
        for fc in self.file_contents:
            parts.append(fc)
        
        # Image analyses next
        for ia in self.image_analyses:
            parts.append(ia)
        
        # Web search LAST - supplementary information only
        if self.web_results:
            parts.append(self.web_results)
        
        return "\n\n".join(parts)


class Orchestrator:
    """
    Main orchestrator that:
    1. Receives queries (with optional files/images)
    2. Routes to appropriate model tier
    3. Gathers context (web search, file reading, image analysis)
    4. Generates response using selected model
    
    All automatic - users just chat.
    """
    
    SYSTEM_PROMPT_ANALYST = """You are an expert document analyst and data specialist. Your PRIMARY purpose is to analyze uploaded documents, spreadsheets, databases, and files.

CRITICAL RULES:
1. ALWAYS base your answers on the actual document content provided in the context
2. NEVER make up or hallucinate information - only state what is IN the documents
3. If document content is provided, your answer MUST reference specific data from it
4. Quote exact values, numbers, names, and text from the documents when relevant
5. If asked about something not in the provided documents, clearly state "This information is not in the provided document(s)"
6. CITATIONS: When referencing external information (web search), YOU MUST use Markdown links: [Source Name](URL). Example: [Wikipedia](https://en.wikipedia.org/...)

When analyzing documents:
- For spreadsheets: Identify column headers, summarize data patterns, provide statistics
- For text documents: Extract key points, summarize sections, identify main themes
- For databases: Describe table structure, summarize record counts, highlight key fields
- Always cite which file/sheet/section your information comes from

You also have web search capabilities for supplementary information, but uploaded documents are ALWAYS the primary source of truth."""

    SYSTEM_PROMPT_ASSISTANT = """You are a highly intelligent, friendly, and conversational AI companion.

Your goal is to have natural, fluid conversations with the user.
- Adopt a warm, professional, yet approachable tone.
- Do not sound robotic or overly formal.
- If the user asks a simple question, give a direct answer without unnecessary preamble.
- You have access to web search and other tools; use them seamlessly when needed.
- Engage with the user's intent, not just their literal words.
- CITATIONS: If you use information from web search results, YOU MUST cite the source using a Markdown link: [Source Name](URL). Do not just list the URL.
"""
    
    def __init__(self):
        self.settings = settings
        self.router = QueryRouter()
        self.web_search = WebSearchTool()
        self.file_reader = FileReaderTool()
        self.vision = VisionTool()
        self.database = DatabaseTool()
        self.rag = RAGTool()  # RAG for intelligent document retrieval
        
        # Conversation history
        self.messages: List[Message] = []
        self.max_history = 20
    
    def process(
        self,
        query: str,
        files: Optional[List[Tuple[str, bytes]]] = None,
        images: Optional[List[Tuple[str, bytes]]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Process a query with optional files and images.
        Yields event dictionaries for streaming.
        
        Args:
            query: User's question/request
            files: List of (filename, bytes) tuples for documents
            images: List of (filename, bytes) tuples for images
        """
        files = files or []
        images = images or []
        
        # Determine file types
        file_types = [Path(f[0]).suffix.lower() for f in files]
        has_files = len(files) > 0
        has_images = len(images) > 0
        
        # Route the query
        decision = self.router.route(
            query=query,
            has_files=has_files,
            file_types=file_types,
            has_images=has_images
        )
        
        # Log routing decision (backend only)
        print(f"\n[Router] {self.router.explain_decision(decision)}")
        
        # Yield routing event
        yield {
            "type": "routing",
            "tier": decision.tier.value,
            "tools": [t.value for t in decision.tools],
            "reasoning": decision.reasoning
        }
        
        # Create processing context
        context = ProcessingContext(query=query, decision=decision)
        
        # Gather context based on routing decision
        # We need to modify _gather_context to yield events too, but for now let's just wrap it
        # or refactor it to be a generator.
        # For simplicity, let's iterate manually here or make _gather_context yield events.
        
        for event in self._gather_context_generator(context, files, images):
            yield event
        
        # Get model config
        model_config = self.settings.get_model_for_tier(decision.tier)
        
        # Build messages for LLM
        llm_messages = self._build_messages(query, context)
        
        # Stream response
        for chunk in self._generate_response(model_config, llm_messages, query, images):
            yield {"type": "content", "content": chunk}
    
    def _gather_context_generator(
        self,
        context: ProcessingContext,
        files: List[Tuple[str, bytes]],
        images: List[Tuple[str, bytes]]
    ) -> Generator[Dict[str, Any], None, None]:
        """Gather all context needed for the query - yielding status events"""
        
        # STEP 1: Process and index uploaded files with RAG
        if files:
            yield {"type": "tool", "status": "running", "tool": "rag_indexing", "message": f"Processing {len(files)} files..."}
            print(f"[Context] Processing {len(files)} uploaded file(s) with RAG...")
            self._process_files_with_rag(context, files)
            self._process_databases(context, files)
            yield {"type": "tool", "status": "complete", "tool": "rag_indexing"}
        
        # STEP 2: Retrieve relevant chunks using RAG
        if files:
            yield {"type": "tool", "status": "running", "tool": "rag_retrieval", "message": "Searching documents..."}
            self._retrieve_with_rag(context)
            yield {"type": "tool", "status": "complete", "tool": "rag_retrieval"}
        
        # STEP 3: Web search if routing decided it's needed
        if ToolType.WEB_SEARCH in context.decision.tools:
            yield {"type": "tool", "status": "running", "tool": "web_search", "message": "Searching the web..."}
            self._do_web_search(context)
            yield {"type": "tool", "status": "complete", "tool": "web_search"}
        
        # STEP 4: Image analysis (for non-vision models that need text description)
        if images and context.decision.tier != ModelTier.VISION:
            yield {"type": "tool", "status": "running", "tool": "vision_analysis", "message": "Analyzing images..."}
            self._analyze_images_as_text(context, images)
            yield {"type": "tool", "status": "complete", "tool": "vision_analysis"}

    def _gather_context(self, *args, **kwargs):
        """Deprecated synchronous wrapper"""
        pass

    
    def _do_web_search(self, context: ProcessingContext):
        """Perform web search"""
        try:
            results = self.web_search.search(context.query)
            context.web_results = self.web_search.format_results(results)
            print(f"[Web Search] Found {len(results)} results")
        except Exception as e:
            print(f"[Web Search] Error: {e}")
    
    def _process_files_with_rag(self, context: ProcessingContext, files: List[Tuple[str, bytes]]):
        """Process document files and index them with RAG"""
        for filename, file_bytes in files:
            ext = Path(filename).suffix.lower()
            
            # Skip database files - handled separately
            if ext in {'.db', '.sqlite', '.sqlite3', '.mdb', '.accdb', '.qvd'}:
                continue
            
            # Skip images - handled separately
            if ext in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}:
                continue
            
            try:
                print(f"[RAG] Reading {filename} ({len(file_bytes)} bytes)...")
                content = self.file_reader.read(filename, file_bytes)
                
                if content.error:
                    print(f"[RAG] ERROR: {content.error}")
                    continue
                    
                if not content.content:
                    print(f"[RAG] WARNING: No content extracted from {filename}")
                    continue
                
                # Determine if structured data
                is_structured = ext in {'.xlsx', '.xls', '.csv', '.json'}
                
                # Index with RAG
                num_chunks = self.rag.add_document(
                    filename=filename,
                    content=content.content,
                    is_structured=is_structured
                )
                print(f"[RAG] Indexed {filename} -> {num_chunks} chunks")
                
            except Exception as e:
                import traceback
                print(f"[RAG] EXCEPTION reading {filename}: {e}")
                traceback.print_exc()
    
    def _retrieve_with_rag(self, context: ProcessingContext):
        """Retrieve relevant chunks using RAG based on the query"""
        try:
            result = self.rag.retrieve(context.query, n_results=15)
            if result.chunks:
                context.rag_context = self.rag.format_context(result)
                print(f"[RAG] Retrieved {len(result.chunks)} relevant chunks from {result.source_files}")
            else:
                print(f"[RAG] No relevant chunks found")
        except Exception as e:
            print(f"[RAG] Retrieval error: {e}")
    
    def _process_databases(self, context: ProcessingContext, files: List[Tuple[str, bytes]]):
        """Process database files"""
        db_extensions = {'.db', '.sqlite', '.sqlite3', '.mdb', '.accdb', '.qvd'}
        
        for filename, file_bytes in files:
            ext = Path(filename).suffix.lower()
            if ext not in db_extensions:
                continue
            
            try:
                content = self.database.read(filename, file_bytes)
                formatted = self.database.format_content(content)
                context.file_contents.append(formatted)
                print(f"[Database] Processed {filename}")
            except Exception as e:
                print(f"[Database] Error reading {filename}: {e}")
    
    def _analyze_images_as_text(self, context: ProcessingContext, images: List[Tuple[str, bytes]]):
        """Analyze images and convert to text descriptions (for non-vision models)"""
        vision_model = self.settings.models.get(ModelTier.VISION)
        if not vision_model:
            return
        
        for filename, image_bytes in images:
            try:
                image_b64 = self.vision.prepare_image(file_bytes=image_bytes)
                if image_b64:
                    analysis = self.vision.analyze_sync(
                        "Describe this image in detail.",
                        image_b64, 
                        model=vision_model.name,
                        filename=filename
                    )
                    formatted = self.vision.format_analysis(analysis)
                    context.image_analyses.append(formatted)
                    print(f"[Vision] Analyzed {filename}")
            except Exception as e:
                print(f"[Vision] Error analyzing {filename}: {e}")
    
    def _build_messages(self, query: str, context: ProcessingContext) -> List[Dict[str, Any]]:
        """Build message list for LLM"""
        # Dynamic System Prompt
        # If we have file content or RAG context, use the strict Analyst prompt
        # Otherwise, use the friendly Assistant prompt
        if context.file_contents or context.rag_context:
            system_prompt = self.SYSTEM_PROMPT_ANALYST
        else:
            system_prompt = self.SYSTEM_PROMPT_ASSISTANT
            
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (limited)
        for msg in self.messages[-self.max_history:]:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Build user message with context
        user_content = query
        context_str = context.build_context()
        
        if context_str:
            user_content = f"""Context information:
{context_str}

User question: {query}"""
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def _generate_response(
        self,
        model_config,
        messages: List[Dict[str, Any]],
        user_query: str,
        images: Optional[List[Tuple[str, bytes]]] = None
    ) -> Generator[str, None, None]:
        """Generate response from model, streaming"""
        
        # Prepare request with full context window support
        request_data = {
            "model": model_config.name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": model_config.temperature,
                "num_predict": model_config.max_tokens,
                "num_ctx": model_config.context_window,  # CRITICAL: Set context window size
                "top_p": model_config.top_p,
                "top_k": model_config.top_k,
                "num_gpu": 99,  # Use all GPU layers
                "num_thread": 24  # Use all CPU threads
            }
        }
        
        # Add images for vision model
        if model_config.supports_vision and images:
            image_data = []
            for filename, img_bytes in images:
                b64 = self.vision.prepare_image(file_bytes=img_bytes)
                if b64:
                    image_data.append(b64)
            if image_data:
                request_data["messages"][-1]["images"] = image_data
        
        print(f"[Model] Using {model_config.name} ({model_config.tier.value})")
        print(f"[Model] Context window: {model_config.context_window:,} tokens, Max output: {model_config.max_tokens:,} tokens")
        
        full_response = ""
        
        try:
            # Long timeout for large models (70b can be slow on first load)
            with httpx.Client(timeout=600.0) as client:
                with client.stream(
                    "POST",
                    "http://localhost:11434/api/chat",
                    json=request_data,
                    timeout=600.0
                ) as response:
                    if response.status_code != 200:
                        yield f"Error: Model returned status {response.status_code}. Make sure '{model_config.name}' is installed in Ollama."
                        return
                    
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    chunk = data["message"]["content"]
                                    full_response += chunk
                                    yield chunk
                            except json.JSONDecodeError:
                                continue
        
        except httpx.ConnectError:
            yield "Error: Cannot connect to Ollama. Make sure Ollama is running."
            return
        except Exception as e:
            yield f"Error generating response: {str(e)}"
            return
        
        # Save to history
        if full_response:
            self.messages.append(Message(role="user", content=user_query))
            self.messages.append(Message(role="assistant", content=full_response))
            
            # Trim history
            if len(self.messages) > self.max_history * 2:
                self.messages = self.messages[-self.max_history * 2:]
    
    def clear_history(self):
        """Clear conversation history"""
        self.messages = []
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available Ollama models"""
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return data.get("models", [])
        except:
            pass
        return []
    
    def check_models(self) -> Dict[ModelTier, bool]:
        """Check which tier models are available"""
        available = self.get_available_models()
        available_names = {m["name"] for m in available}
        
        status = {}
        for tier, config in self.settings.models.items():
            name = config.name
            base_name = name.split(":")[0]
            status[tier] = (
                name in available_names or 
                f"{base_name}:latest" in available_names or
                any(base_name in m for m in available_names)
            )
        
        return status
