"""Orchestrator - the brain of the system.

Coordinates routing + tools + generation.
All orchestration happens on the backend; the UI is just a client.
"""
from typing import List, Optional, Generator, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import httpx
import json
import time

from config import settings, ModelTier, ToolType
from config.admin_overrides import get_model_overrides, get_system_prompt, get_tool_override
from core.router import QueryRouter, RoutingDecision
from core.agents import get_agent
from tools import (
    WebSearchTool,
    FileReaderTool,
    VisionTool,
    DatabaseTool,
    RAGTool,
    SpreadsheetTool,
    CalculatorTool,
    MicrosoftGraphTool,
)
from tools.calculator import looks_like_math_query
from tools.ms_graph import extract_microsoft_urls

logger = logging.getLogger(__name__)


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
        self.spreadsheet = SpreadsheetTool()
        self.calculator = CalculatorTool()
        self.ms_graph = MicrosoftGraphTool()
        
        # Conversation history
        self.messages: List[Message] = []
        self.max_history = 20
    
    def process(
        self,
        query: str,
        files: Optional[List[Tuple[str, bytes]]] = None,
        images: Optional[List[Tuple[str, bytes]]] = None,
        user: Optional[Dict[str, Any]] = None,
        token: Optional[Dict[str, Any]] = None,
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

        # Per-request identity context (used for permission-aware tools).
        # Not persisted beyond this request.
        self._request_user = user or {}
        self._request_token = token or {}
        
        # Determine file types
        file_types = [Path(f[0]).suffix.lower() for f in files]
        has_files = len(files) > 0
        has_images = len(images) > 0
        
        try:
            # Route the query
            decision = self.router.route(
                query=query,
                has_files=has_files,
                file_types=file_types,
                has_images=has_images,
            )

            # Log routing decision (backend only)
            logger.info("[Router] %s", self.router.explain_decision(decision))

            # Yield routing event
            yield {
                "type": "routing",
                "tier": decision.tier.value,
                "tools": [t.value for t in decision.tools],
                "reasoning": decision.reasoning,
            }

            # Create processing context
            context = ProcessingContext(query=query, decision=decision)

            # Gather context based on routing decision
            for event in self._gather_context_generator(context, files, images):
                yield event

            # Get model config (supports admin-managed runtime overrides)
            model_config = self._get_effective_model_config(decision.tier)

            # Build messages for LLM
            llm_messages = self._build_messages(query, context)

            # Stream response
            for event in self._generate_response(model_config, llm_messages, query, images):
                yield event
        finally:
            self._request_user = {}
            self._request_token = {}

    def process_agent(
        self,
        agent_name: str,
        query: str,
        files: Optional[List[Tuple[str, bytes]]] = None,
        images: Optional[List[Tuple[str, bytes]]] = None,
        user: Optional[Dict[str, Any]] = None,
        token: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Run a named workforce agent to complete a task.

        This is similar to process(), but uses a fixed tool/tier profile and an
        additional system preamble to drive task-completion behavior.
        """

        files = files or []
        images = images or []

        self._request_user = user or {}
        self._request_token = token or {}

        try:
            agent = get_agent(agent_name)

            decision = RoutingDecision(
                tier=agent.tier,
                tools=agent.tools,
                reasoning=f"Agent '{agent.name}' selected",
            )

            yield {
                "type": "routing",
                "tier": decision.tier.value,
                "tools": [t.value for t in decision.tools],
                "reasoning": decision.reasoning,
                "agent": agent.name,
            }

            context = ProcessingContext(query=query, decision=decision)
            for event in self._gather_context_generator(context, files, images):
                yield event

            model_config = self._get_effective_model_config(decision.tier)
            llm_messages = self._build_messages(query, context)

            # Prepend agent-specific system preamble.
            llm_messages.insert(0, {"role": "system", "content": agent.system_preamble})

            for event in self._generate_response(model_config, llm_messages, query, images):
                yield event
        finally:
            self._request_user = {}
            self._request_token = {}
    
    def _gather_context_generator(
        self,
        context: ProcessingContext,
        files: List[Tuple[str, bytes]],
        images: List[Tuple[str, bytes]]
    ) -> Generator[Dict[str, Any], None, None]:
        """Gather all context needed for the query - yielding status events"""

        # STEP 0: Permission-aware Microsoft reads (SharePoint/OneDrive links)
        # Uses delegated Graph token for the signed-in user, so access is enforced by Microsoft.
        ms_urls = extract_microsoft_urls(context.query)
        access_token = (getattr(self, "_request_token", {}) or {}).get("access_token")
        if ms_urls and access_token:
            max_bytes = int(getattr(self.settings, "max_file_size_mb", 100)) * 1024 * 1024
            # Avoid pulling too many external files per prompt.
            for url in ms_urls[:2]:
                yield {
                    "type": "tool",
                    "status": "running",
                    "tool": "microsoft_read",
                    "message": "Reading Microsoft link...",
                }
                try:
                    ref = self.ms_graph.resolve_share_link(share_url=url, access_token=access_token)
                    data = self.ms_graph.download_drive_item_content(
                        drive_id=ref.drive_id,
                        item_id=ref.item_id,
                        access_token=access_token,
                        max_bytes=max_bytes,
                    )

                    filename = ref.name or "microsoft_file"
                    # Treat as if the user uploaded it: extract + index.
                    self._process_files_with_rag(context, [(filename, data)])
                    yield {"type": "tool", "status": "complete", "tool": "microsoft_read"}
                except Exception:
                    logger.exception("[Microsoft] Failed to read link")
                    yield {"type": "tool", "status": "error", "tool": "microsoft_read", "message": "Microsoft read failed."}
        
        # STEP 1: Process and index uploaded files with RAG
        if files:
            yield {"type": "tool", "status": "running", "tool": "rag_indexing", "message": f"Processing {len(files)} files..."}
            logger.debug("[Context] Processing %s uploaded file(s) with RAG...", len(files))
            self._process_files_with_rag(context, files)
            self._process_databases(context, files)
            yield {"type": "tool", "status": "complete", "tool": "rag_indexing"}

        # STEP 1.25: Calculator (local math) for computation questions
        calculator_enabled_override = get_tool_override("calculator_enabled")
        calculator_enabled = (
            bool(calculator_enabled_override)
            if isinstance(calculator_enabled_override, bool)
            else bool(getattr(self.settings, "calculator_enabled", True))
        )
        should_calc = (
            ToolType.CALCULATOR in (context.decision.tools or [])
            or looks_like_math_query(context.query)
        )

        if calculator_enabled and should_calc:
            yield {"type": "tool", "status": "running", "tool": "calculator", "message": "Computing..."}
            try:
                self._run_calculator(context)
                yield {"type": "tool", "status": "complete", "tool": "calculator"}
            except Exception as e:
                yield {
                    "type": "error",
                    "where": "server",
                    "retryable": False,
                    "message": f"Calculator failed: {str(e)}",
                }
                yield {"type": "tool", "status": "error", "tool": "calculator", "message": "Calculator failed."}
        elif should_calc and not calculator_enabled:
            yield {
                "type": "tool",
                "status": "skipped",
                "tool": "calculator",
                "message": "Calculator is disabled by administrator.",
            }

        # STEP 1.5: Spreadsheet computation for structured files
        if files:
            spreadsheet_enabled_override = get_tool_override("spreadsheet_enabled")
            spreadsheet_enabled = (
                bool(spreadsheet_enabled_override)
                if isinstance(spreadsheet_enabled_override, bool)
                else bool(getattr(self.settings, "spreadsheet_enabled", True))
            )

            structured_files = [
                (fn, b)
                for (fn, b) in files
                if Path(fn).suffix.lower() in {".xlsx", ".xls", ".csv"}
            ]

            if structured_files and spreadsheet_enabled:
                yield {"type": "tool", "status": "running", "tool": "spreadsheet", "message": "Computing spreadsheet results..."}
                try:
                    self._process_spreadsheets(context, structured_files)
                    yield {"type": "tool", "status": "complete", "tool": "spreadsheet"}
                except Exception as e:
                    yield {
                        "type": "error",
                        "where": "server",
                        "retryable": False,
                        "message": f"Spreadsheet processing failed: {str(e)}",
                    }
                    yield {"type": "tool", "status": "error", "tool": "spreadsheet", "message": "Spreadsheet processing failed."}
            elif structured_files and not spreadsheet_enabled:
                yield {
                    "type": "tool",
                    "status": "skipped",
                    "tool": "spreadsheet",
                    "message": "Spreadsheet compute is disabled by administrator.",
                }
        
        # STEP 2: Retrieve relevant chunks using RAG
        # If docs were previously indexed in this session, retrieve on every turn.
        if files or self.rag.has_documents():
            yield {"type": "tool", "status": "running", "tool": "rag_retrieval", "message": "Searching documents..."}
            result = self._retrieve_with_rag(context)
            rag_sources = self._build_rag_sources(result)
            if rag_sources:
                yield {"type": "sources", "sources": rag_sources, "merge": True}
            yield {"type": "tool", "status": "complete", "tool": "rag_retrieval"}
        
        # STEP 3: Web search if routing decided it's needed (and not globally disabled)
        web_search_enabled_override = get_tool_override("web_search_enabled")
        web_search_enabled = (
            bool(web_search_enabled_override)
            if isinstance(web_search_enabled_override, bool)
            else True
        )

        if ToolType.WEB_SEARCH in context.decision.tools and web_search_enabled:
            yield {"type": "tool", "status": "running", "tool": "web_search", "message": "Searching the web..."}
            results = self._do_web_search(context)
            web_sources = self._build_web_sources(results)
            if web_sources:
                yield {"type": "sources", "sources": web_sources, "merge": True}
            yield {"type": "tool", "status": "complete", "tool": "web_search"}
        elif ToolType.WEB_SEARCH in context.decision.tools and not web_search_enabled:
            yield {
                "type": "tool",
                "status": "skipped",
                "tool": "web_search",
                "message": "Web search is disabled by administrator."
            }
        
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
            logger.debug("[Web Search] Found %s results", len(results))
            return results
        except Exception as e:
            logger.exception("[Web Search] Error")
            return []
    
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
                logger.debug("[RAG] Reading %s (%s bytes)...", filename, len(file_bytes))
                content = self.file_reader.read(filename, file_bytes)
                
                if content.error:
                    logger.warning("[RAG] Read error for %s: %s", filename, content.error)
                    continue
                    
                if not content.content:
                    logger.warning("[RAG] No content extracted from %s", filename)
                    continue
                
                # Determine if structured data
                is_structured = ext in {'.xlsx', '.xls', '.csv', '.json'}
                
                # Index with RAG
                num_chunks = self.rag.add_document(
                    filename=filename,
                    content=content.content,
                    is_structured=is_structured
                )
                logger.debug("[RAG] Indexed %s -> %s chunks", filename, num_chunks)
                
            except Exception as e:
                logger.exception("[RAG] Exception reading/indexing %s", filename)
    
    def _retrieve_with_rag(self, context: ProcessingContext):
        """Retrieve relevant chunks using RAG based on the query"""
        try:
            result = self.rag.retrieve(context.query, n_results=15)
            if result.chunks:
                context.rag_context = self.rag.format_context(result)
                logger.debug("[RAG] Retrieved %s relevant chunks from %s", len(result.chunks), result.source_files)
            else:
                logger.debug("[RAG] No relevant chunks found")
            return result
        except Exception as e:
            logger.exception("[RAG] Retrieval error")
            return None

    def _build_web_sources(self, results) -> List[Dict[str, Any]]:
        """Normalize web search results into UI-friendly sources."""
        sources: List[Dict[str, Any]] = []
        for r in (results or []):
            url = getattr(r, "url", "") or ""
            if not url:
                continue
            title = (getattr(r, "title", "") or "").strip()
            snippet = (getattr(r, "snippet", "") or "").strip()
            src = (getattr(r, "source", "") or "").strip()
            sources.append(
                {
                    "kind": "web",
                    "href": url,
                    "label": title or url,
                    "snippet": snippet,
                    "source": src,
                }
            )
        return sources

    def _build_rag_sources(self, result) -> List[Dict[str, Any]]:
        """Normalize RAG retrieval results into UI-friendly sources."""
        sources: List[Dict[str, Any]] = []
        if not result or not getattr(result, "chunks", None):
            return sources

        # De-dupe by (filename, chunk_num) while preserving order.
        seen = set()
        for chunk in result.chunks:
            meta = getattr(chunk, "metadata", {}) or {}
            filename = meta.get("filename") or "document"
            chunk_num = meta.get("chunk_num")
            key = (filename, chunk_num)
            if key in seen:
                continue
            seen.add(key)
            label = f"{filename}"
            if chunk_num is not None:
                label = f"{filename} (section {chunk_num})"
            sources.append(
                {
                    "kind": "document",
                    "href": "",
                    "label": label,
                    "filename": filename,
                    "chunk_id": getattr(chunk, "chunk_id", ""),
                    "chunk_num": chunk_num,
                }
            )

        return sources
    
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
                logger.debug("[Database] Processed %s", filename)
            except Exception as e:
                logger.exception("[Database] Error reading %s", filename)
    
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
                    logger.debug("[Vision] Analyzed %s", filename)
            except Exception as e:
                logger.exception("[Vision] Error analyzing %s", filename)

    def _run_calculator(self, context: ProcessingContext) -> None:
        """Plan a safe expression and evaluate it locally."""

        # If the user provided a bare expression, try it directly first.
        direct = (context.query or "").strip()
        if direct and len(direct) <= int(getattr(self.settings, "calculator_max_expression_length", 512) or 512):
            try:
                result = self.calculator.evaluate(direct)
                context.file_contents.append(self.calculator.format_result(result))
                return
            except Exception:
                pass

        # Otherwise ask the model to translate the question into an expression.
        model_config = self._get_effective_model_config(context.decision.tier)
        plan = self._plan_calculator_expression(model_config, context.query)
        expr = (plan.get("expression") or "").strip()
        if not expr:
            # Nothing to compute.
            return

        result = self.calculator.evaluate(expr)
        rationale = (plan.get("rationale") or plan.get("explanation") or "").strip()
        if rationale:
            context.file_contents.append(f"[Calculator Plan]\n{rationale}")
        context.file_contents.append(self.calculator.format_result(result))

    def _plan_calculator_expression(self, model_config, user_query: str) -> Dict[str, Any]:
        """Ask the model for a safe math expression we can evaluate locally."""
        system = (
            "You translate user math questions into a single safe expression. Return JSON only.\n\n"
            "Rules:\n"
            "- Output a single JSON object with keys: expression, rationale\n"
            "- expression MUST be a single math expression compatible with Python math syntax\n"
            "- Do NOT include code, imports, attributes, indexing, variables other than: pi, e, tau\n"
            "- Allowed functions: abs, round, min, max, sum, sqrt, log, log10, exp, sin, cos, tan, asin, acos, atan, degrees, radians, floor, ceil, factorial, comb, perm\n"
            "- If the question needs external facts (exchange rates, unit conversions you don't know), return an empty expression and explain in rationale.\n"
        )
        user = f"Question: {user_query}"

        content = self._ollama_chat_once(
            model=model_config.name,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options={
                "temperature": 0.0,
                "num_predict": 256,
                "num_ctx": min(int(getattr(model_config, "context_window", 4096) or 4096), 4096),
                "top_p": 1.0,
                "top_k": 20,
            },
        )

        parsed = self._extract_json_object(content)
        if not isinstance(parsed, dict):
            return {"expression": "", "rationale": "Planner returned non-JSON output."}

        expr = (parsed.get("expression") or "").strip()
        if expr and ";" in expr:
            return {"expression": "", "rationale": "Unsafe expression (multiple statements)."}

        return parsed

    def _process_spreadsheets(self, context: ProcessingContext, files: List[Tuple[str, bytes]]):
        """Profile and (when useful) compute answers over structured spreadsheet/CSV uploads."""
        # Load tables from all structured files.
        tables: Dict[str, Any] = {}
        profiles = []

        for filename, file_bytes in files:
            file_tables, file_profiles = self.spreadsheet.load_tables_from_bytes(filename, file_bytes)

            # Merge with unique table names.
            for name, df in file_tables.items():
                unique = name
                i = 2
                while unique in tables:
                    unique = f"{name}_{i}"
                    i += 1
                tables[unique] = df

                # Update profile table_name if renamed.
                for p in file_profiles:
                    if p.table_name == name:
                        p.table_name = unique
                        break

            profiles.extend(file_profiles)

        if not profiles:
            return

        # Always add a compact schema/profile to context.
        context.file_contents.append(self.spreadsheet.format_profiles(profiles))

        # Add deterministic, value-level profiling to improve reliability.
        # This helps the model pick the correct columns/values (e.g., Completed vs Closed - Completed).
        try:
            context.file_contents.append(self.spreadsheet.format_table_insights(tables, profiles))
        except Exception:
            logger.exception("[Spreadsheet] Failed to build table insights")

        # Only attempt query planning/execution when it looks like the user wants computation,
        # or the router selected the spreadsheet tool.
        wants_compute = (
            ToolType.SPREADSHEET in (context.decision.tools or [])
            or self._query_looks_like_spreadsheet_compute(context.query)
        )
        if not wants_compute:
            return

        # Deterministic fallback for a very common "easy" case: incident counts by person/month.
        # This avoids brittle LLM-authored SQL selecting the wrong columns or using overly strict equality.
        try:
            incident = self.spreadsheet.try_answer_incident_count(context.query, tables)
        except Exception:
            incident = None

        if incident is not None:
            cols = incident.get("columns_used") or {}
            context.file_contents.append(
                "[Spreadsheet Fact]\n"
                f"Incidents completed by {incident.get('person')} in {incident.get('month'):02d}/{incident.get('year')}: {incident.get('count')}\n"
                f"(table: {incident.get('table')}; columns: status='{cols.get('status')}', person='{cols.get('person')}', date='{cols.get('date')}')"
            )
            return

        model_config = self._get_effective_model_config(context.decision.tier)
        try:
            planner_hints = self.spreadsheet.build_planner_hints(tables, profiles)
        except Exception:
            planner_hints = ""

        plan = self._plan_spreadsheet_sql(model_config, context.query, profiles, planner_hints)
        sql = (plan.get("sql") or "").strip()
        if not sql:
            # If the planner couldn't produce a safe/meaningful query, fall back to profiles only.
            return

        result = self.spreadsheet.run_query(tables, sql)
        formatted = self.spreadsheet.format_query_result(result)
        rationale = (plan.get("rationale") or plan.get("explanation") or "").strip()
        if rationale:
            context.file_contents.append(f"[Spreadsheet Plan]\n{rationale}")
        context.file_contents.append(formatted)

    @staticmethod
    def _query_looks_like_spreadsheet_compute(query: str) -> bool:
        q = (query or "").lower()
        keywords = [
            "pivot",
            "group by",
            "sum",
            "average",
            "mean",
            "median",
            "min",
            "max",
            "count",
            "distinct",
            "trend",
            "correlation",
            "regression",
            "dedupe",
            "deduplicate",
            "join",
            "lookup",
            "vlookup",
            "match",
            "top ",
            "bottom ",
        ]
        return any(k in q for k in keywords)

    def _plan_spreadsheet_sql(self, model_config, user_query: str, profiles, planner_hints: str = "") -> Dict[str, Any]:
        """Ask the model to produce a SELECT-only DuckDB SQL query for the user's request."""

        schema_lines = []
        for p in profiles:
            cols = ", ".join(p.columns[:80])
            schema_lines.append(f"- {p.table_name} (rows≈{p.rows}, cols={p.cols}) columns: {cols}")
        schema = "\n".join(schema_lines)

        system = (
            "You are a data analyst writing DuckDB SQL over uploaded tables. "
            "Return JSON only.\n\n"
            "Rules:\n"
            "- Output a single JSON object with keys: sql, rationale\n"
            "- sql MUST be a single SELECT statement (or WITH ... SELECT).\n"
            "- Do NOT use semicolons.\n"
            "- Do NOT use any write/DDL operations (CREATE, INSERT, UPDATE, DELETE, COPY, ATTACH, PRAGMA, INSTALL, LOAD, etc).\n"
            "- If the question cannot be answered from the available columns, set sql to an empty string and explain why in rationale.\n"
        )

        user = (
            f"Available tables and columns:\n{schema}\n\n"
            + (f"Value-level profiling hints (deterministic):\n{planner_hints}\n\n" if planner_hints else "")
            + f"User question:\n{user_query}\n"
        )

        content = self._ollama_chat_once(
            model=model_config.name,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options={
                "temperature": 0.0,
                "num_predict": 512,
                "num_ctx": min(int(getattr(model_config, "context_window", 4096) or 4096), 8192),
                "top_p": 1.0,
                "top_k": 20,
            },
        )

        parsed = self._extract_json_object(content)
        if not isinstance(parsed, dict):
            return {"sql": "", "rationale": "Planner returned non-JSON output."}

        sql = (parsed.get("sql") or "").strip()
        if sql:
            ok, reason = self.spreadsheet.is_safe_select_sql(sql)
            if not ok:
                return {"sql": "", "rationale": f"Planner produced unsafe SQL ({reason})."}
        return parsed

    def _ollama_chat_once(self, model: str, messages: List[Dict[str, Any]], options: Dict[str, Any]) -> str:
        """Non-streaming Ollama chat call for internal planning steps."""

        base_url = getattr(self.settings, "ollama_base_url", "http://localhost:11434").rstrip("/")
        url = f"{base_url}/api/chat"
        retries = int(getattr(self.settings, "ollama_retries", 2) or 0)
        backoff = float(getattr(self.settings, "ollama_retry_backoff_seconds", 0.5) or 0.0)

        timeout = httpx.Timeout(
            connect=float(getattr(self.settings, "ollama_connect_timeout_seconds", 5.0)),
            read=float(getattr(self.settings, "ollama_read_timeout_seconds", 600.0)),
            write=float(getattr(self.settings, "ollama_write_timeout_seconds", 30.0)),
            pool=5.0,
        )

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        for attempt in range(retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    msg = (data.get("message") or {}).get("content")
                    return msg or ""

                if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(backoff * (2**attempt))
                    continue

                raise RuntimeError(f"Ollama planning call failed ({resp.status_code})")
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
                if attempt < retries:
                    time.sleep(backoff * (2**attempt))
                    continue
                raise

        return ""

    @staticmethod
    def _extract_json_object(text: str) -> Any:
        """Best-effort JSON object extraction from model output."""
        if not text:
            return None
        t = text.strip()
        try:
            return json.loads(t)
        except Exception:
            pass

        # Try to extract first {...} block.
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = t[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                return None
        return None
    
    def _build_messages(self, query: str, context: ProcessingContext) -> List[Dict[str, Any]]:
        """Build message list for LLM"""
        # Dynamic System Prompt
        # If we have file content or RAG context, use the strict Analyst prompt
        # Otherwise, use the friendly Assistant prompt
        analyst_mode = bool(context.file_contents or context.rag_context)
        system_prompt = get_system_prompt(
            analyst=analyst_mode,
            default_assistant=self.SYSTEM_PROMPT_ASSISTANT,
            default_analyst=self.SYSTEM_PROMPT_ANALYST,
        )
            
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

    def _get_effective_model_config(self, tier: ModelTier):
        """Return a ModelConfig with admin overrides applied (without mutating global settings)."""
        base = self.settings.get_model_for_tier(tier)
        overrides = get_model_overrides().get(tier.value)
        if not overrides:
            return base

        # Only allow overriding safe, expected fields.
        allowed_keys = {
            "name",
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "context_window",
        }
        safe_updates = {k: v for k, v in overrides.items() if k in allowed_keys}
        if not safe_updates:
            return base

        merged = base.model_dump()
        merged.update(safe_updates)
        # Ensure tier stays consistent
        merged["tier"] = tier
        return type(base)(**merged)
    
    def _generate_response(
        self,
        model_config,
        messages: List[Dict[str, Any]],
        user_query: str,
        images: Optional[List[Tuple[str, bytes]]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate response from model, streaming."""
        
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
        
        logger.info("[Model] Using %s (%s)", model_config.name, model_config.tier.value)
        logger.debug(
            "[Model] Context window: %s tokens, Max output: %s tokens",
            f"{model_config.context_window:,}",
            f"{model_config.max_tokens:,}",
        )
        
        full_response = ""
        
        base_url = getattr(self.settings, "ollama_base_url", "http://localhost:11434").rstrip("/")
        url = f"{base_url}/api/chat"

        retries = int(getattr(self.settings, "ollama_retries", 2) or 0)
        backoff = float(getattr(self.settings, "ollama_retry_backoff_seconds", 0.5) or 0.0)

        timeout = httpx.Timeout(
            connect=float(getattr(self.settings, "ollama_connect_timeout_seconds", 5.0)),
            read=float(getattr(self.settings, "ollama_read_timeout_seconds", 600.0)),
            write=float(getattr(self.settings, "ollama_write_timeout_seconds", 30.0)),
            pool=5.0,
        )

        attempt = 0
        while True:
            try:
                with httpx.Client(timeout=timeout) as client:
                    with client.stream("POST", url, json=request_data) as response:
                        if response.status_code != 200:
                            yield {
                                "type": "error",
                                "where": "ollama",
                                "retryable": False,
                                "message": (
                                    f"Model returned status {response.status_code}. "
                                    f"Make sure '{model_config.name}' is installed and Ollama is healthy."
                                ),
                            }
                            return

                        for line in response.iter_lines():
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            if "message" in data and "content" in data["message"]:
                                chunk = data["message"]["content"]
                                if chunk:
                                    full_response += chunk
                                    yield {"type": "content", "content": chunk}

                break

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                if attempt < retries:
                    time.sleep(backoff * (2**attempt))
                    attempt += 1
                    continue

                if isinstance(e, httpx.ConnectError):
                    yield {
                        "type": "error",
                        "where": "ollama",
                        "retryable": True,
                        "message": "Cannot connect to Ollama. Make sure Ollama is running.",
                    }
                else:
                    yield {
                        "type": "error",
                        "where": "ollama",
                        "retryable": True,
                        "message": f"Ollama request failed: {str(e)}",
                    }
                return
            except Exception as e:
                yield {
                    "type": "error",
                    "where": "ollama",
                    "retryable": False,
                    "message": f"Error generating response: {str(e)}",
                }
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
        base_url = getattr(self.settings, "ollama_base_url", "http://localhost:11434").rstrip("/")
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return data.get("models", [])
        except Exception:
            logger.exception("Failed to fetch available Ollama models")
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
