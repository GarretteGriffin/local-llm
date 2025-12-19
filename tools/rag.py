"""
RAG (Retrieval Augmented Generation) Tool
Uses ChromaDB for vector storage and Ollama for embeddings.
All local - no API keys needed.
"""
import hashlib
import httpx
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
import re

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk of text from a document"""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str


@dataclass
class RetrievalResult:
    """Results from RAG retrieval"""
    chunks: List[Chunk]
    total_chars: int
    source_files: List[str]


class RAGTool:
    """
    Retrieval Augmented Generation for document analysis.
    
    Instead of stuffing entire documents into context, we:
    1. Chunk documents into smaller pieces
    2. Create embeddings using Ollama's nomic-embed-text
    3. Store in ChromaDB (local vector database)
    4. Retrieve only relevant chunks based on query
    
    This allows handling documents of ANY size efficiently.
    """
    
    CHUNK_SIZE = 1000  # ~250 tokens per chunk
    CHUNK_OVERLAP = 200  # Overlap for context continuity
    MAX_CHUNKS_TO_RETRIEVE = 10  # Top K chunks to return
    EMBEDDING_MODEL = "nomic-embed-text"
    
    def __init__(self, persist_dir: Optional[str] = None):
        """Initialize RAG with optional persistence directory"""
        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            # In-memory for session-based use
            self.client = chromadb.Client(ChromaSettings(
                anonymized_telemetry=False
            ))
        
        # Create or get collection for current session
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        self._session_docs: Dict[str, str] = {}  # Track docs added this session

    def has_documents(self) -> bool:
        """Return True if any documents have been indexed this session."""
        return bool(self._session_docs)
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from Ollama"""
        base_url = settings.ollama_base_url.rstrip("/")
        url = f"{base_url}/api/embeddings"
        retries = int(getattr(settings, "ollama_retries", 2) or 0)
        backoff = float(getattr(settings, "ollama_retry_backoff_seconds", 0.5) or 0.0)

        timeout = httpx.Timeout(
            connect=float(getattr(settings, "ollama_connect_timeout_seconds", 5.0)),
            read=30.0,
            write=float(getattr(settings, "ollama_write_timeout_seconds", 30.0)),
            pool=5.0,
        )

        for attempt in range(retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(
                        url,
                        json={
                            "model": self.EMBEDDING_MODEL,
                            "prompt": text,
                        },
                    )

                if response.status_code == 200:
                    return response.json().get("embedding")

                # Retry on transient upstream errors.
                if response.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(backoff * (2**attempt))
                    continue

                logger.warning("RAG embedding failed (%s)", response.status_code)
                return None
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                if attempt < retries:
                    time.sleep(backoff * (2**attempt))
                    continue
                logger.exception("RAG embedding connection error")
                return None
            except Exception:
                logger.exception("RAG embedding error")
                return None
        return None
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Get embeddings for multiple texts (aligned to input order)."""
        if not texts:
            return []

        # Ollama's embeddings endpoint is per-prompt; parallelize calls for throughput.
        max_workers = min(8, (os.cpu_count() or 4))
        results: List[Optional[List[float]]] = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._get_embedding, t): i for i, t in enumerate(texts)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception:
                    logger.exception("RAG embedding worker failed")
                    results[idx] = None

        return results
    
    def _chunk_text(self, text: str, filename: str) -> List[Chunk]:
        """Split text into overlapping chunks"""
        chunks = []
        
        # Clean the text
        text = text.strip()
        if not text:
            return chunks
        
        # Split by paragraphs first, then by size
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        chunk_num = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds chunk size, save current and start new
            if len(current_chunk) + len(para) > self.CHUNK_SIZE and current_chunk:
                chunk_id = f"{hashlib.md5(filename.encode()).hexdigest()[:8]}_{chunk_num}"
                chunks.append(Chunk(
                    text=current_chunk.strip(),
                    metadata={"filename": filename, "chunk_num": chunk_num},
                    chunk_id=chunk_id
                ))
                chunk_num += 1
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-self.CHUNK_OVERLAP//5:] if len(words) > self.CHUNK_OVERLAP//5 else words
                current_chunk = " ".join(overlap_words) + "\n\n" + para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunk_id = f"{hashlib.md5(filename.encode()).hexdigest()[:8]}_{chunk_num}"
            chunks.append(Chunk(
                text=current_chunk.strip(),
                metadata={"filename": filename, "chunk_num": chunk_num},
                chunk_id=chunk_id
            ))
        
        return chunks
    
    def _chunk_structured_data(self, text: str, filename: str) -> List[Chunk]:
        """Chunk structured data (spreadsheets, databases) differently"""
        chunks = []
        lines = text.split('\n')
        
        # First line is usually header
        header = lines[0] if lines else ""
        
        current_chunk = header + "\n"
        chunk_num = 0
        rows_in_chunk = 0
        MAX_ROWS_PER_CHUNK = 50  # Keep chunks manageable
        
        for line in lines[1:]:
            if not line.strip():
                continue
            
            current_chunk += line + "\n"
            rows_in_chunk += 1
            
            # Create chunk when we hit row limit or size limit
            if rows_in_chunk >= MAX_ROWS_PER_CHUNK or len(current_chunk) > self.CHUNK_SIZE:
                chunk_id = f"{hashlib.md5(filename.encode()).hexdigest()[:8]}_{chunk_num}"
                chunks.append(Chunk(
                    text=current_chunk.strip(),
                    metadata={"filename": filename, "chunk_num": chunk_num, "type": "structured"},
                    chunk_id=chunk_id
                ))
                chunk_num += 1
                
                # Start new chunk with header
                current_chunk = header + "\n"
                rows_in_chunk = 0
        
        # Last chunk
        if rows_in_chunk > 0:
            chunk_id = f"{hashlib.md5(filename.encode()).hexdigest()[:8]}_{chunk_num}"
            chunks.append(Chunk(
                text=current_chunk.strip(),
                metadata={"filename": filename, "chunk_num": chunk_num, "type": "structured"},
                chunk_id=chunk_id
            ))
        
        return chunks
    
    def add_document(self, filename: str, content: str, is_structured: bool = False) -> int:
        """
        Add a document to the RAG index.
        
        Args:
            filename: Name of the file
            content: Full text content
            is_structured: True for spreadsheets/databases
            
        Returns:
            Number of chunks created
        """
        # Check if already indexed
        doc_hash = hashlib.md5(content.encode()).hexdigest()
        if filename in self._session_docs and self._session_docs[filename] == doc_hash:
            logger.info("RAG: %s already indexed; skipping", filename)
            return 0
        
        # Remove old chunks for this file if re-indexing
        try:
            existing = self.collection.get(where={"filename": filename})
            if existing and existing.get('ids'):
                self.collection.delete(ids=existing['ids'])
        except Exception:
            logger.exception("RAG: failed to delete existing chunks for %s", filename)
        
        # Chunk the document
        if is_structured:
            chunks = self._chunk_structured_data(content, filename)
        else:
            chunks = self._chunk_text(content, filename)
        
        if not chunks:
            return 0
        
        logger.info("RAG: chunking %s into %d chunks", filename, len(chunks))
        
        # Get embeddings (parallel). Skip chunks that fail embedding rather than inserting
        # meaningless zero vectors.
        texts = [c.text for c in chunks]
        max_workers = min(8, (os.cpu_count() or 4))
        embedded: List[Tuple[Chunk, List[float]]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._get_embedding, t): i for i, t in enumerate(texts)}
            for fut in as_completed(futures):
                idx = futures[fut]
                emb = None
                try:
                    emb = fut.result()
                except Exception:
                    logger.exception("RAG: embedding worker failed")
                if emb:
                    embedded.append((chunks[idx], emb))
                else:
                    logger.warning("RAG: skipping chunk %s (embedding failed)", chunks[idx].chunk_id)

        if not embedded:
            logger.warning("RAG: no chunks embedded for %s; skipping index", filename)
            return 0

        # Keep stable order by chunk_num for readability/debuggability
        embedded.sort(key=lambda pair: pair[0].metadata.get("chunk_num", 0))

        self.collection.add(
            ids=[c.chunk_id for c, _ in embedded],
            embeddings=[e for _, e in embedded],
            documents=[c.text for c, _ in embedded],
            metadatas=[c.metadata for c, _ in embedded],
        )
        
        self._session_docs[filename] = doc_hash
        logger.info("RAG: indexed %d chunks from %s", len(embedded), filename)
        
        return len(embedded)
    
    def retrieve(self, query: str, n_results: int = None) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: The user's question
            n_results: Number of chunks to retrieve (default: MAX_CHUNKS_TO_RETRIEVE)
            
        Returns:
            RetrievalResult with relevant chunks
        """
        n_results = n_results or self.MAX_CHUNKS_TO_RETRIEVE
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            logger.warning("RAG: failed to get query embedding")
            return RetrievalResult(chunks=[], total_chars=0, source_files=[])
        
        # Query ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.exception("RAG query error")
            return RetrievalResult(chunks=[], total_chars=0, source_files=[])
        
        # Build result
        chunks = []
        source_files = set()
        total_chars = 0
        
        if results and results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0
                
                # Only include if similarity is good enough (distance < 0.8 for cosine)
                if distance < 0.8:
                    chunks.append(Chunk(
                        text=doc,
                        metadata=metadata,
                        chunk_id=results['ids'][0][i]
                    ))
                    source_files.add(metadata.get('filename', 'unknown'))
                    total_chars += len(doc)
        
        logger.debug("RAG retrieved %s chunks (%s chars)", len(chunks), f"{total_chars:,}")
        
        return RetrievalResult(
            chunks=chunks,
            total_chars=total_chars,
            source_files=list(source_files)
        )
    
    def format_context(self, result: RetrievalResult) -> str:
        """Format retrieved chunks for LLM context"""
        if not result.chunks:
            return ""
        
        parts = [f"[Retrieved from: {', '.join(result.source_files)}]"]
        
        for i, chunk in enumerate(result.chunks, 1):
            filename = chunk.metadata.get('filename', 'document')
            chunk_num = chunk.metadata.get('chunk_num', 0)
            parts.append(f"\n--- Chunk {i} (from {filename}, section {chunk_num}) ---")
            parts.append(chunk.text)
        
        return "\n".join(parts)
    
    def clear(self):
        """Clear all indexed documents"""
        try:
            self.client.delete_collection("documents")
            self.collection = self.client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            self._session_docs.clear()
            logger.info("RAG cleared all indexed documents")
        except Exception as e:
            logger.exception("RAG clear error")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed documents"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "indexed_files": list(self._session_docs.keys()),
                "embedding_model": self.EMBEDDING_MODEL
            }
        except Exception:
            logger.exception("RAG stats error")
            return {"total_chunks": 0, "indexed_files": [], "embedding_model": self.EMBEDDING_MODEL}
