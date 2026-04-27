"""
Document pipeline: loads text/PDF files, chunks them, and generates embeddings
using a local sentence-transformers model (no API key required).
"""
from pathlib import Path
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def get_embeddings() -> HuggingFaceEmbeddings:
    """Returns a cached HuggingFace embedding model."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents(policies_path: Path) -> List[Document]:
    """Loads all .txt and .pdf files from the policies directory."""
    docs: List[Document] = []
    for path in policies_path.glob("**/*"):
        if path.suffix == ".txt":
            try:
                loader = TextLoader(str(path), encoding="utf-8")
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source"] = path.name
                docs.extend(loaded)
                logger.info(f"Loaded: {path.name}")
            except Exception as e:
                logger.error(f"Failed to load {path.name}: {e}")
        elif path.suffix == ".pdf":
            try:
                loader = PyPDFLoader(str(path))
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source"] = path.name
                docs.extend(loaded)
                logger.info(f"Loaded PDF: {path.name} ({len(loaded)} pages)")
            except Exception as e:
                logger.error(f"Failed to load PDF {path.name}: {e}")

    logger.info(f"Total documents loaded: {len(docs)}")
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Splits documents into overlapping chunks for dense retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")
    return chunks


def load_and_chunk(policies_path: Path) -> Tuple[List[Document], int]:
    """Full pipeline: load → chunk → return chunks and doc count."""
    docs = load_documents(policies_path)
    chunks = chunk_documents(docs)
    return chunks, len(docs)
