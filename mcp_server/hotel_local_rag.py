# hotel_local_rag.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document



class HotelLocalRAG:
    def __init__(
        self,
        vector_db_path: str = "./hotel_vector_db",
        embedding_model_name: str = "all-MiniLM-L6-v2",  # Free local model
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.vector_db_path = Path(vector_db_path)
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vector_db: Optional[FAISS] = None
        self.documents_metadata: Dict = {}

        
        self._initialize_components()
        self._load_existing_db()

    def _initialize_components(self) -> None:
        """Khởi tạo text splitter & local embeddings (HuggingFace)."""
        try:
            # Text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )

            # Local embedding model (HuggingFace sentence-transformers)
            print(f"🔄 Loading local embedding model: {self.embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},  # Use CPU to avoid GPU issues
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"✅ Using Local HuggingFace Embeddings: {self.embedding_model_name}")

        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            self.embeddings = None

    def initialize_if_needed(self):
        """Initialize vector DB if it doesn't exist"""
        if not os.path.exists(self.vector_db_path):
            print("Initializing vector database...", file=sys.stderr)
            # Tạo empty vector DB hoặc load từ documents có sẵn
            return True
        return False

    def _load_existing_db(self) -> None:
        """Load existing FAISS index nếu đã tồn tại."""
        try:
            if self.vector_db_path.exists() and self.embeddings:
                self.vector_db = FAISS.load_local(
                    str(self.vector_db_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )

                # Load metadata
                metadata_file = self.vector_db_path / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        self.documents_metadata = json.load(f)

                print(f"✅ Loaded existing vector database from {self.vector_db_path}")
        except Exception as e:
            print(f"⚠️  Could not load existing database: {e}")

    def create_db_from_txt_folder(self, txt_folder_path: str) -> Dict:
        """Tạo/cập nhật database từ thư mục chứa file TXT."""
        if not self.embeddings:
            return {"error": "Local embeddings không khả dụng"}

        txt_folder = Path(txt_folder_path)
        if not txt_folder.exists():
            return {"error": f"Thư mục không tồn tại: {txt_folder_path}"}

        all_documents: List[Document] = []
        processed_files: List[str] = []

        print(f"📁 Processing TXT files in {txt_folder_path}")

        for txt_file in txt_folder.glob("*.txt"):
            try:
                print(f"📄 Processing: {txt_file.name}")
                loader = TextLoader(str(txt_file), encoding="utf-8")
                documents = loader.load()

                # Thêm metadata & split
                for doc in documents:
                    doc.metadata.update(
                        {
                            "source_file": txt_file.name,
                            "file_type": "txt",
                            "category": "hotel-docs",
                            "processed_date": datetime.now().isoformat(),
                        }
                    )
                chunks = self.text_splitter.split_documents(documents)
                all_documents.extend(chunks)
                processed_files.append(txt_file.name)
            except Exception as e:
                print(f"❌ Error processing {txt_file.name}: {e}")

        if not all_documents:
            return {"error": "Không có documents nào được xử lý thành công"}

        try:
            print(f"🔄 Creating embeddings for {len(all_documents)} chunks...")
            if self.vector_db is None:
                self.vector_db = FAISS.from_documents(all_documents, self.embeddings)
            else:
                new_db = FAISS.from_documents(all_documents, self.embeddings)
                self.vector_db.merge_from(new_db)

            self.vector_db.save_local(str(self.vector_db_path))

            # Cập nhật metadata
            self.documents_metadata.update(
                {
                    "txt_folder": txt_folder_path,
                    "processed_files": processed_files,
                    "total_chunks": len(all_documents),
                    "last_updated": datetime.now().isoformat(),
                    "embedding_model": self.embedding_model_name,
                }
            )
            self._save_metadata()

            return {
                "success": True,
                "processed_files": processed_files,
                "total_chunks": len(all_documents),
                "vector_db_path": str(self.vector_db_path),
            }
        except Exception as e:
            return {"error": f"Lỗi tạo vector database: {e}"}

    def create_db_from_pdf_folder(self, pdf_folder_path: str) -> Dict:
        """Tạo/cập nhật database từ thư mục chứa file PDF."""
        if  not self.embeddings:
            return {"error": "Local embeddings không khả dụng"}

        pdf_folder = Path(pdf_folder_path)
        if not pdf_folder.exists():
            return {"error": f"Thư mục không tồn tại: {pdf_folder_path}"}

        all_documents: List[Document] = []
        processed_files: List[str] = []

        print(f"📁 Processing PDF files in {pdf_folder_path}")

        for pdf_file in pdf_folder.glob("*.pdf"):
            try:
                print(f"📄 Processing: {pdf_file.name}")
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()

                # Thêm metadata & split
                for doc in documents:
                    doc.metadata.update(
                        {
                            "source_file": pdf_file.name,
                            "file_type": "pdf",
                            "category": "hotel-docs",
                            "processed_date": datetime.now().isoformat(),
                        }
                    )
                chunks = self.text_splitter.split_documents(documents)
                all_documents.extend(chunks)
                processed_files.append(pdf_file.name)
            except Exception as e:
                print(f"❌ Error processing {pdf_file.name}: {e}")

        if not all_documents:
            return {"error": "Không có documents nào được xử lý thành công"}

        try:
            print(f"🔄 Creating embeddings for {len(all_documents)} chunks...")
            if self.vector_db is None:
                self.vector_db = FAISS.from_documents(all_documents, self.embeddings)
            else:
                new_db = FAISS.from_documents(all_documents, self.embeddings)
                self.vector_db.merge_from(new_db)

            self.vector_db.save_local(str(self.vector_db_path))

            # Cập nhật metadata
            meta = self.documents_metadata
            meta.setdefault("processed_files", [])
            meta["processed_files"].extend(processed_files)
            meta.update(
                {
                    "pdf_folder": pdf_folder_path,
                    "total_chunks": meta.get("total_chunks", 0) + len(all_documents),
                    "last_updated": datetime.now().isoformat(),
                    "embedding_model": self.embedding_model_name,
                }
            )
            self._save_metadata()

            return {
                "success": True,
                "processed_files": processed_files,
                "total_chunks": len(all_documents),
                "vector_db_path": str(self.vector_db_path),
            }
        except Exception as e:
            return {"error": f"Lỗi tạo vector database: {e}"}

    def query(self, question: str, k: int = 3, score_threshold: float = 0.5) -> List[Dict]:
        """Truy vấn FAISS index và trả kết quả chuẩn hoá điểm."""
        if not self.vector_db:
            return [{"error": "Vector database chưa được tạo"}]

        try:
            results = self.vector_db.similarity_search_with_score(question, k=k)

            formatted_results: List[Dict] = []
            for doc, distance in results:
                # FAISS distance -> similarity score (higher is better)
                similarity = 1.0 / (1.0 + distance) if distance > 0 else 1.0
                if similarity >= score_threshold:
                    formatted_results.append(
                        {
                            "text": doc.page_content,
                            "score": round(float(similarity), 3),
                            "source_file": doc.metadata.get("source_file", "Unknown"),
                            "file_type": doc.metadata.get("file_type", "Unknown"),
                            "category": doc.metadata.get("category", "Unknown"),
                            "page": doc.metadata.get("page", None),
                        }
                    )

            return formatted_results
        except Exception as e:
            return [{"error": f"Lỗi query: {e}"}]

    def _save_metadata(self) -> None:
        try:
            self.vector_db_path.mkdir(exist_ok=True)
            metadata_file = self.vector_db_path / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.documents_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save metadata: {e}")

    def get_summary(self) -> Dict:
        if not self.vector_db:
            return {"error": "Vector database chưa được tạo"}
        return {
            "status": "ready",
            "vector_db_path": str(self.vector_db_path),
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "documents_metadata": self.documents_metadata,
        }

    def is_ready(self) -> bool:
        return self.vector_db is not None


def setup_hotel_rag(
    txt_folder: Optional[str] = None,
    pdf_folder: Optional[str] = None,
    vector_db_path: str = "./hotel_vector_db",
    embedding_model: str = "all-MiniLM-L6-v2",
):
    """Setup Hotel RAG system with local embeddings"""
    print(" Setting up Ohana Hotel RAG System (Local Embeddings - FREE)")
    print("=" * 60)

    rag = HotelLocalRAG(
        vector_db_path=vector_db_path,
        embedding_model_name=embedding_model,
    )

    if not rag.embeddings:
        print(" Cannot initialize local embedding model")
        return None

    results: List[tuple[str, Dict]] = []

    if txt_folder and os.path.exists(txt_folder):
        print(f"\nProcessing TXT folder: {txt_folder}")
        results.append(("TXT", rag.create_db_from_txt_folder(txt_folder)))

    if pdf_folder and os.path.exists(pdf_folder):
        print(f"\n Processing PDF folder: {pdf_folder}")
        results.append(("PDF", rag.create_db_from_pdf_folder(pdf_folder)))

    print("\nSETUP RESULTS:")
    print("-" * 30)
    for file_type, result in results:
        if result.get("success"):
            print(f" {file_type}: {len(result['processed_files'])} files, {result['total_chunks']} chunks")
        else:
            print(f" {file_type}: {result.get('error', 'Unknown error')}")

    if rag.is_ready():
        print("\nTesting query... (ví dụ: 'nội quy khách sạn')")
        test_results = rag.query("nội quy khách sạn", k=2)
        for i, r in enumerate(test_results, 1):
            if "error" not in r:
                print(f"{i}. Score: {r['score']:.3f} | {r['source_file']}")
                print(f"   {r['text'][:100]}...")

    return rag




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup Hotel RAG System (Local Embeddings - FREE)"
    )

    parser.add_argument(
        "--pdf-folder",
        type=Path,
        default=Path("./Data"),
        metavar="PDF_DIR",
        help='Thư mục .pdf (ví dụ: "Dịch vụ")'
    )
    parser.add_argument(
        "--vector-db",
        type=Path,
        default=Path("./hotel_vector_db"),
        metavar="DB_DIR",
        help="Thư mục lưu FAISS index (mặc định: ./hotel_vector_db)"
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Tên mô hình embedding local (mặc định: all-MiniLM-L6-v2)"
    )

    args = parser.parse_args()

  
    pdf_folder = str(args.pdf_folder.resolve()) if args.pdf_folder and args.pdf_folder.exists() else None
    vector_db = str(args.vector_db.resolve())

    if not pdf_folder:
        print("❌ Specify at least --txt-folder or --pdf-folder")
        sys.exit(1)

    setup_hotel_rag(
        pdf_folder=pdf_folder,
        vector_db_path=vector_db,
        embedding_model=args.embedding_model,
    )