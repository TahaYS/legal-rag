"""
Legal RAG — FastAPI Backend
Serves the RAG pipeline as a REST API.

Endpoints:
    POST /query      — Ask a legal question, get a cited answer
    GET  /health     — Check if the API is running
    GET  /stats      — Get collection stats (chunk count, etc.)

Usage:
    uvicorn src.api.app:app --reload
"""

import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.rag.rag_pipeline import LegalRAGPipeline


# ── Request/Response Models ─────────────────────────────────────

class QueryRequest(BaseModel):
    """What the frontend sends to us."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Legal question about the Pakistan Penal Code",
        examples=["What is the punishment for theft?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of source chunks to retrieve",
    )


class Source(BaseModel):
    """A single source citation."""
    section: str
    title: str
    page: int
    similarity: float
    preview: str


class QueryResponse(BaseModel):
    """What we send back to the frontend."""
    question: str
    answer: str
    sources: list[Source]
    time_taken: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    chunks_loaded: int
    model: str


# ── App Initialization ──────────────────────────────────────────

# Store the pipeline globally so it loads once, not per request
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the RAG pipeline when the server starts.
    This runs once — not on every request.
    Loading the embedding model and ChromaDB takes a few seconds.
    """
    global pipeline
    print("\nStarting Legal RAG API server...")
    pipeline = LegalRAGPipeline()
    print("Server ready!\n")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Legal RAG API",
    description="AI-powered Pakistan Penal Code assistant with cited answers",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow frontend to connect (CORS)
# In production, replace "*" with your actual frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ───────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_legal(request: QueryRequest):
    """
    Ask a legal question about the Pakistan Penal Code.
    Returns an AI-generated answer with cited PPC sections.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")

    start = time.time()

    try:
        result = pipeline.query(
            question=request.question,
            top_k=request.top_k,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}",
        )

    elapsed = round(time.time() - start, 2)

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[
            Source(
                section=s["section"],
                title=s["title"],
                page=s["page"],
                similarity=s["similarity"],
                preview=s["preview"],
            )
            for s in result["sources"]
        ],
        time_taken=elapsed,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and the pipeline is loaded."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    return HealthResponse(
        status="healthy",
        chunks_loaded=pipeline.retriever.collection.count(),
        model="llama-3.1-8b-instant",
    )


@app.get("/stats")
async def get_stats():
    """Get statistics about the loaded legal knowledge base."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    count = pipeline.retriever.collection.count()
    return {
        "collection": "pakistan_penal_code",
        "total_chunks": count,
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "llama-3.1-8b-instant",
        "top_k_default": 5,
    }


# ── Run directly ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
