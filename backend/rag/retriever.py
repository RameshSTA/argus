"""
FAISS vector store: build, persist, load, and search.
Manages the dense retrieval index over policy documents.
"""
from pathlib import Path
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from backend.rag.embedder import get_embeddings, load_and_chunk
from backend.rag.schemas import IndexStatus
from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self):
        self._store: Optional[FAISS] = None
        self._doc_count: int = 0
        self._chunk_count: int = 0

    def build(self, policies_path: Path) -> IndexStatus:
        """Build FAISS index from policy documents."""
        settings = get_settings()
        chunks, doc_count = load_and_chunk(policies_path)

        if not chunks:
            raise ValueError("No documents found to index.")

        embeddings = get_embeddings()
        self._store = FAISS.from_documents(chunks, embeddings)
        self._doc_count = doc_count
        self._chunk_count = len(chunks)

        # Persist index
        index_path = settings.faiss_full_path
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(index_path))
        logger.info(f"FAISS index saved → {index_path} ({len(chunks)} vectors)")

        return IndexStatus(
            status="built",
            document_count=doc_count,
            chunk_count=len(chunks),
            index_path=str(index_path),
        )

    def load(self) -> bool:
        """Load persisted FAISS index from disk."""
        settings = get_settings()
        index_path = settings.faiss_full_path

        if not index_path.exists():
            return False

        try:
            embeddings = get_embeddings()
            self._store = FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"FAISS index loaded from {index_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False

    def ensure_loaded(self) -> None:
        """Load from disk, or rebuild from documents if not found."""
        if self._store is not None:
            return
        if not self.load():
            logger.info("No saved index found — building from documents")
            settings = get_settings()
            self.build(settings.policies_full_path)

    def search(self, query: str, top_k: int = 4) -> List[Document]:
        """Similarity search over the vector store."""
        self.ensure_loaded()
        return self._store.similarity_search(query, k=top_k)

    def search_with_scores(self, query: str, top_k: int = 4) -> List[tuple[Document, float]]:
        """Search with relevance scores (lower = more similar for L2)."""
        self.ensure_loaded()
        return self._store.similarity_search_with_score(query, k=top_k)

    @property
    def is_ready(self) -> bool:
        return self._store is not None

    def status(self) -> IndexStatus:
        settings = get_settings()
        return IndexStatus(
            status="ready" if self.is_ready else "not_loaded",
            document_count=self._doc_count,
            chunk_count=self._chunk_count,
            index_path=str(settings.faiss_full_path),
        )


_vector_store = VectorStore()


def get_vector_store() -> VectorStore:
    return _vector_store
