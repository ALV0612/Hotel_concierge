# hotel_local_rag_optimized.py
from __future__ import annotations

import os
import sys
import gc
import psutil
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime
import threading
import weakref

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

class MemoryMonitor:
    """Memory monitoring and cleanup"""
    
    @staticmethod
    def get_memory_usage():
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    @staticmethod
    def force_cleanup():
        gc.collect()
        
    @staticmethod
    def check_memory_limit(limit_mb=400):  # 80% of 512MB
        current = MemoryMonitor.get_memory_usage()
        if current > limit_mb:
            MemoryMonitor.force_cleanup()
            return True
        return False

class LazyVectorStore:
    """Lazy loading vector store to save memory"""
    
    def __init__(self, db_path: str, embeddings):
        self.db_path = db_path
        self.embeddings = embeddings
        self._db = None
        self._last_used = None
        self._lock = threading.Lock()
    
    @property
    def db(self):
        with self._lock:
            if self._db is None and os.path.exists(self.db_path):
                print(f"🔄 Loading vector DB from {self.db_path}")
                self._db = FAISS.load_local(
                    self.db_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            self._last_used = datetime.now()
            return self._db
    
    def search(self, query: str, k: int = 3):
        db = self.db
        if db is None:
            return []
        return db.similarity_search_with_score(query, k=k)
    
    def save(self, documents: List[Document]):
        """Create or merge documents into vector store"""
        if self._db is None:
            self._db = FAISS.from_documents(documents, self.embeddings)
        else:
            new_db = FAISS.from_documents(documents, self.embeddings)
            self._db.merge_from(new_db)
            del new_db  # Free memory immediately
        
        self._db.save_local(self.db_path)
        MemoryMonitor.force_cleanup()
    
    def unload(self):
        """Manually unload from memory"""
        with self._lock:
            if self._db is not None:
                del self._db
                self._db = None
                MemoryMonitor.force_cleanup()

class OptimizedEmbeddings:
    """Memory-optimized embeddings with smaller model"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._embeddings = None
        
    @property
    def embeddings(self):
        if self._embeddings is None:
            print(f"🔄 Loading embedding model: {self.model_name}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': False,
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 8,  # Smaller batch size
                },
                cache_folder="./models_cache"  # Cache models locally
            )
        return self._embeddings

class HotelLocalRAG:
    def __init__(
        self,
        vector_db_path: str = "./hotel_vector_db",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 256,  # Smaller chunks
        chunk_overlap: int = 32,  # Less overlap
        max_cache_size: int = 100,  # Limit cache
    ):
        self.vector_db_path = Path(vector_db_path)
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components lazily
        self._text_splitter = None
        self._embeddings_wrapper = None
        self._vector_store = None
        self.documents_metadata = {}
        
        # Query cache with LRU
        self._query_cache = {}
        self._cache_order = []
        self.max_cache_size = max_cache_size
        
        self._load_metadata()

    @property
    def text_splitter(self):
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        return self._text_splitter

    @property
    def embeddings(self):
        if self._embeddings_wrapper is None:
            self._embeddings_wrapper = OptimizedEmbeddings(self.embedding_model_name)
        return self._embeddings_wrapper.embeddings

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = LazyVectorStore(
                str(self.vector_db_path), 
                self.embeddings
            )
        return self._vector_store

    def _load_metadata(self):
        """Load metadata without loading vector DB"""
        try:
            metadata_file = self.vector_db_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    self.documents_metadata = json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load metadata: {e}")

    def _save_metadata(self):
        try:
            self.vector_db_path.mkdir(exist_ok=True)
            metadata_file = self.vector_db_path / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.documents_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save metadata: {e}")

    def _update_cache(self, query: str, results: List[Dict]):
        """LRU cache for query results"""
        if query in self._query_cache:
            self._cache_order.remove(query)
        elif len(self._query_cache) >= self.max_cache_size:
            # Remove oldest
            oldest = self._cache_order.pop(0)
            del self._query_cache[oldest]
        
        self._query_cache[query] = results
        self._cache_order.append(query)

    def process_documents_streaming(self, file_paths: List[str], batch_size: int = 5):
        """Process documents in batches to save memory"""
        all_chunks = []
        processed_files = []
        
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i + batch_size]
            batch_documents = []
            
            print(f"📄 Processing batch {i//batch_size + 1}: {len(batch_files)} files")
            
            for file_path in batch_files:
                try:
                    file_path = Path(file_path)
                    
                    if file_path.suffix.lower() == '.pdf':
                        loader = PyPDFLoader(str(file_path))
                    else:
                        loader = TextLoader(str(file_path), encoding="utf-8")
                    
                    documents = loader.load()
                    
                    for doc in documents:
                        doc.metadata.update({
                            "source_file": file_path.name,
                            "file_type": file_path.suffix[1:],
                            "category": "hotel-docs",
                            "processed_date": datetime.now().isoformat(),
                        })
                    
                    chunks = self.text_splitter.split_documents(documents)
                    batch_documents.extend(chunks)
                    processed_files.append(file_path.name)
                    
                    # Memory check after each file
                    if MemoryMonitor.check_memory_limit():
                        print("⚠️ Memory limit reached, forcing cleanup")
                        
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")
            
            if batch_documents:
                all_chunks.extend(batch_documents)
            
            # Clear batch from memory
            del batch_documents
            MemoryMonitor.force_cleanup()
        
        return all_chunks, processed_files

    def create_db_from_folder(self, folder_path: str, file_types: List[str] = ['.txt', '.pdf']) -> Dict:
        """Create/update database from folder with memory optimization"""
        folder = Path(folder_path)
        if not folder.exists():
            return {"error": f"Folder not found: {folder_path}"}

        # Find all files
        all_files = []
        for file_type in file_types:
            all_files.extend(folder.glob(f"*{file_type}"))
        
        if not all_files:
            return {"error": "No supported files found"}

        try:
            # Process in batches
            all_documents, processed_files = self.process_documents_streaming(
                [str(f) for f in all_files], 
                batch_size=3  # Very small batches for 512MB
            )
            
            if not all_documents:
                return {"error": "No documents processed successfully"}

            print(f"🔄 Creating embeddings for {len(all_documents)} chunks...")
            
            # Save to vector store
            self.vector_store.save(all_documents)
            
            # Update metadata
            self.documents_metadata.update({
                "folder_path": folder_path,
                "processed_files": processed_files,
                "total_chunks": len(all_documents),
                "last_updated": datetime.now().isoformat(),
                "embedding_model": self.embedding_model_name,
            })
            self._save_metadata()
            
            # Clear from memory
            del all_documents
            MemoryMonitor.force_cleanup()

            return {
                "success": True,
                "processed_files": processed_files,
                "total_chunks": len(processed_files),
                "vector_db_path": str(self.vector_db_path),
            }
            
        except Exception as e:
            return {"error": f"Error creating vector database: {e}"}

    def query(self, question: str, k: int = 3, score_threshold: float = 0.5) -> List[Dict]:
        """Query with caching and memory optimization"""
        
        # Check cache first
        if question in self._query_cache:
            return self._query_cache[question]
        
        try:
            results = self.vector_store.search(question, k=k)
            
            formatted_results = []
            for doc, distance in results:
                # FAISS distance -> similarity score
                similarity = 1.0 / (1.0 + distance) if distance > 0 else 1.0
                if similarity >= score_threshold:
                    formatted_results.append({
                        "text": doc.page_content,
                        "score": round(float(similarity), 3),
                        "source_file": doc.metadata.get("source_file", "Unknown"),
                        "file_type": doc.metadata.get("file_type", "Unknown"),
                        "category": doc.metadata.get("category", "Unknown"),
                        "page": doc.metadata.get("page", None),
                    })
            
            # Cache results
            self._update_cache(question, formatted_results)
            
            # Memory cleanup
            if MemoryMonitor.check_memory_limit():
                self.cleanup_memory()
            
            return formatted_results
            
        except Exception as e:
            return [{"error": f"Query error: {e}"}]

    def cleanup_memory(self):
        """Force memory cleanup"""
        print("🧹 Cleaning up memory...")
        
        # Clear query cache
        self._query_cache.clear()
        self._cache_order.clear()
        
        # Unload vector store
        if self._vector_store:
            self._vector_store.unload()
        
        # Force garbage collection
        MemoryMonitor.force_cleanup()
        
        print(f"✅ Memory usage: {MemoryMonitor.get_memory_usage():.1f}MB")

    def get_summary(self) -> Dict:
        memory_usage = MemoryMonitor.get_memory_usage()
        
        return {
            "status": "ready" if self.is_ready() else "not_ready",
            "vector_db_path": str(self.vector_db_path),
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "memory_usage_mb": round(memory_usage, 1),
            "cache_size": len(self._query_cache),
            "documents_metadata": self.documents_metadata,
        }

    def is_ready(self) -> bool:
        return self.vector_db_path.exists()

# Usage example with auto-cleanup
def setup_optimized_hotel_rag(
    folder_path: str,
    vector_db_path: str = "./hotel_vector_db",
    embedding_model: str = "all-MiniLM-L6-v2",
):
    """Setup optimized RAG for 512MB environment"""
    print("🏨 Setting up Optimized Hotel RAG System")
    print("=" * 50)
    
    # Monitor initial memory
    initial_memory = MemoryMonitor.get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.1f}MB")
    
    rag = HotelLocalRAG(
        vector_db_path=vector_db_path,
        embedding_model_name=embedding_model,
        chunk_size=256,  # Smaller chunks
        chunk_overlap=32,
    )
    
    if folder_path and os.path.exists(folder_path):
        result = rag.create_db_from_folder(folder_path)
        
        if result.get("success"):
            print(f"✅ Processed: {len(result['processed_files'])} files")
            print(f"📊 Final memory: {MemoryMonitor.get_memory_usage():.1f}MB")
        else:
            print(f"❌ Error: {result.get('error')}")
    
    return rag

if __name__ == "__main__":
    # Example usage
    rag = setup_optimized_hotel_rag("./Data")
    
    if rag.is_ready():
        # Test query
        results = rag.query("nội quy khách sạn", k=2)
        for i, r in enumerate(results, 1):
            if "error" not in r:
                print(f"{i}. Score: {r['score']:.3f} | {r['source_file']}")
                print(f"   {r['text'][:100]}...")