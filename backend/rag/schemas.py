from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500, description="Natural language question")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of document chunks to retrieve")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Does comprehensive motor cover include storm damage to a parked vehicle?",
                "top_k": 4,
            }
        }


class SourceChunk(BaseModel):
    source: str
    page: Optional[int] = None
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    confidence: str  # HIGH | MEDIUM | LOW
    retrieved_chunks: int


class IndexStatus(BaseModel):
    status: str
    document_count: int
    chunk_count: int
    index_path: str
