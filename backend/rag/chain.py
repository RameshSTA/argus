"""
RAG chain: retrieves relevant policy chunks and generates grounded answers
using Claude as the LLM. Falls back to a rule-based extractor when no API key.
"""
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

from backend.rag.retriever import get_vector_store
from backend.rag.schemas import QueryRequest, QueryResponse, SourceChunk
from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

RAG_SYSTEM_PROMPT = """You are an expert insurance policy analyst. Your role is to answer questions
about insurance coverage based solely on the provided policy document excerpts.

Rules:
- Answer only from the provided context. Do not use outside knowledge.
- Be precise and cite specific sections or clauses when available.
- If the context does not contain enough information, say so clearly.
- Use plain, professional language suitable for both customers and claims staff.
- Highlight key conditions, exclusions, and excess amounts where relevant.
"""

RAG_HUMAN_PROMPT = """Policy document excerpts:
{context}

Question: {question}

Provide a clear, accurate answer based strictly on the above excerpts."""


def _build_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        parts.append(f"[Excerpt {i} — {source}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def _extract_confidence(docs: list[Document], question: str) -> str:
    """Heuristic confidence based on retrieved chunk relevance."""
    question_lower = question.lower()
    key_terms = question_lower.split()
    matches = sum(
        1 for doc in docs
        if any(term in doc.page_content.lower() for term in key_terms if len(term) > 4)
    )
    if matches >= 3:
        return "HIGH"
    elif matches >= 1:
        return "MEDIUM"
    return "LOW"


def _rule_based_answer(question: str, docs: list[Document]) -> str:
    """Fallback extractor when no Anthropic API key is configured."""
    context = _build_context(docs)
    q_lower = question.lower()

    # Try to find the most relevant sentence in retrieved chunks
    sentences = []
    for doc in docs:
        sentences.extend([s.strip() for s in doc.page_content.split(".") if len(s) > 40])

    scored = []
    q_words = set(w for w in q_lower.split() if len(w) > 3)
    for sent in sentences:
        score = sum(1 for w in q_words if w in sent.lower())
        if score > 0:
            scored.append((score, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        best = scored[0][1].strip()
        return (
            f"Based on the policy documents: {best}. "
            f"Please review the full policy excerpts for complete details. "
            f"(Note: AI-generated response — configure ANTHROPIC_API_KEY for enhanced answers.)"
        )

    return (
        "The policy documents contain relevant information about this topic. "
        "Please review the source excerpts above for the specific details. "
        "(Configure ANTHROPIC_API_KEY for AI-generated answers.)"
    )


def query(request: QueryRequest) -> QueryResponse:
    settings = get_settings()
    store = get_vector_store()

    docs_with_scores = store.search_with_scores(request.question, top_k=request.top_k)
    docs = [d for d, _ in docs_with_scores]

    if not docs:
        return QueryResponse(
            answer="No relevant policy information found for your query.",
            sources=[],
            confidence="LOW",
            retrieved_chunks=0,
        )

    sources = [
        SourceChunk(
            source=doc.metadata.get("source", "Policy Document"),
            page=doc.metadata.get("page"),
            excerpt=doc.page_content[:300].strip() + "...",
        )
        for doc in docs
    ]

    # Use Claude if API key is available
    if settings.anthropic_api_key:
        try:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model="claude-haiku-4-5-20251001",
                api_key=settings.anthropic_api_key,
                max_tokens=512,
                temperature=0.1,
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", RAG_SYSTEM_PROMPT),
                ("human", RAG_HUMAN_PROMPT),
            ])
            chain = prompt | llm
            context = _build_context(docs)
            response = chain.invoke({"context": context, "question": request.question})
            answer = response.content
        except Exception as e:
            logger.error(f"LLM chain error: {e}")
            answer = _rule_based_answer(request.question, docs)
    else:
        logger.warning("ANTHROPIC_API_KEY not set — using rule-based extraction")
        answer = _rule_based_answer(request.question, docs)

    return QueryResponse(
        answer=answer,
        sources=sources,
        confidence=_extract_confidence(docs, request.question),
        retrieved_chunks=len(docs),
    )
