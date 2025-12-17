"""
RAG (Retrieval Augmented Generation) Tool
Uses ChromaDB for vector storage and Ollama for embeddings.
All local - no API keys needed.
"""
import hashlib
import httpx
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
import re


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
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama"""
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    "http://localhost:11434/api/embeddings",
                    json={
                        "model": self.EMBEDDING_MODEL,
                        "prompt": text
                    }
                )
                if response.status_code == 200:
                    return response.json()["embedding"]
        except Exception as e:
            print(f"[RAG] Embedding error: {e}")
        return []
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            emb = self._get_embedding(text)
            if emb:
                embeddings.append(emb)
            else:
                # Fallback: zero vector (will have low similarity)
                embeddings.append([0.0] * 768)
        return embeddings
    
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
            print(f"[RAG] {filename} already indexed, skipping")
            return 0
        
        # Remove old chunks for this file if re-indexing
        try:
            existing = self.collection.get(where={"filename": filename})
            if existing and existing['ids']:
                self.collection.delete(ids=existing['ids'])
        except:
            pass
        
        # Chunk the document
        if is_structured:
            chunks = self._chunk_structured_data(content, filename)
        else:
            chunks = self._chunk_text(content, filename)
        
        if not chunks:
            return 0
        
        print(f"[RAG] Chunking {filename} into {len(chunks)} chunks...")
        
        # Get embeddings
        texts = [c.text for c in chunks]
        embeddings = self._get_embeddings_batch(texts)
        
        # Add to ChromaDB
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c.metadata for c in chunks]
        )
        
        self._session_docs[filename] = doc_hash
        print(f"[RAG] Indexed {len(chunks)} chunks from {filename}")
        
        return len(chunks)
    
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
            print("[RAG] Failed to get query embedding")
            return RetrievalResult(chunks=[], total_chars=0, source_files=[])
        
        # Query ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"[RAG] Query error: {e}")
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
        
        print(f"[RAG] Retrieved {len(chunks)} relevant chunks ({total_chars:,} chars)")
        
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
            print("[RAG] Cleared all indexed documents")
        except Exception as e:
            print(f"[RAG] Clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed documents"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "indexed_files": list(self._session_docs.keys()),
                "embedding_model": self.EMBEDDING_MODEL
            }
        except:
            return {"total_chunks": 0, "indexed_files": [], "embedding_model": self.EMBEDDING_MODEL}
