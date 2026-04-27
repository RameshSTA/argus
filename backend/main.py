"""
Argus Insurance Intelligence Platform — FastAPI Application
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.config import get_settings, Settings
from backend.utils.logger import setup_logger, get_logger
from backend.ml.schemas import ClaimFeatures, ScoreResponse, TrainResponse
from backend.ml.model import get_inference_engine
from backend.rag.schemas import QueryRequest, QueryResponse, IndexStatus
from backend.rag.retriever import get_vector_store
from backend.rag.chain import query as rag_query
from backend.agent.schemas import AgentRequest, AgentResponse
from backend.agent.agent import run_agent

setup_logger()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: pre-load models and build vector index."""
    settings = get_settings()
    logger.info(f"Starting Argus v{settings.app_version}")

    # Pre-load ML model
    engine = get_inference_engine()
    engine.ensure_loaded()
    logger.info("ML inference engine ready")

    # Pre-load / build RAG index
    store = get_vector_store()
    store.ensure_loaded()
    logger.info("Vector store ready")

    yield

    logger.info("Argus shutting down")


app = FastAPI(
    title="Argus Insurance Intelligence Platform",
    description="ML fraud detection + RAG policy assistant + autonomous claims agent",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ──────────────────────────────────────────────────────────
@app.get("/api/health", tags=["System"])
async def health():
    engine = get_inference_engine()
    store = get_vector_store()
    return {
        "status": "healthy",
        "version": settings.app_version,
        "ml_model": "loaded" if engine._model is not None else "not_loaded",
        "vector_store": "ready" if store.is_ready else "not_ready",
    }


# ── ML Risk Scoring ──────────────────────────────────────────────────
@app.post("/api/score", response_model=ScoreResponse, tags=["ML"])
async def score_claim(claim: ClaimFeatures):
    """Score a claim for fraud probability with SHAP feature attribution."""
    try:
        engine = get_inference_engine()
        return engine.predict(claim)
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train", response_model=TrainResponse, tags=["ML"])
async def train_model():
    """Re-train the XGBoost model on the latest dataset."""
    try:
        from backend.ml.train import train
        return train()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── RAG Policy Assistant ────────────────────────────────────────────
@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
async def query_policy(request: QueryRequest):
    """Query insurance policy documents using RAG."""
    try:
        return rag_query(request)
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/index/status", response_model=IndexStatus, tags=["RAG"])
async def index_status():
    """Get the current vector index status."""
    store = get_vector_store()
    return store.status()


@app.post("/api/index/rebuild", response_model=IndexStatus, tags=["RAG"])
async def rebuild_index():
    """Rebuild the FAISS index from policy documents."""
    try:
        settings = get_settings()
        store = get_vector_store()
        store._store = None
        return store.build(settings.policies_full_path)
    except Exception as e:
        logger.error(f"Index rebuild error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Claims Agent ────────────────────────────────────────────────────
@app.post("/api/agent", response_model=AgentResponse, tags=["Agent"])
async def run_claims_agent(request: AgentRequest):
    """Run the autonomous claims intelligence agent."""
    try:
        return run_agent(request)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Frontend ────────────────────────────────────────────────────────
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    static_path = frontend_path / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        index = frontend_path / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return JSONResponse({"message": "Argus API is running. Visit /api/docs"})

    @app.get("/{full_path:path}", include_in_schema=False)
    async def catch_all(full_path: str):
        file_path = frontend_path / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        index = frontend_path / "index.html"
        if index.exists():
            return FileResponse(str(index))
        raise HTTPException(status_code=404)
