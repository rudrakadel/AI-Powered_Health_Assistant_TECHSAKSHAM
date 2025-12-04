"""
FastAPI application entry point.
Defines API endpoints for medical query processing.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import redis
from celery.result import AsyncResult

from app.config import settings
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    TaskStatus,
    HealthResponse
)
from app.tasks.celery_tasks import query_task, celery_app
from app.rag.retriever import RAGRetriever
from app.utils.logger import get_logger
from app.utils.metrics import metrics_tracker, QueryMetrics
from datetime import datetime

logger = get_logger(__name__)

# Initialize RAG on startup
rag_retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global rag_retriever
    
    # Startup
    logger.info("Initializing Medical AI Assistant")
    
    try:
        # Initialize RAG system
        logger.info("Initializing RAG retriever...")
        rag_retriever = RAGRetriever()
        rag_retriever.initialize()
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG: {str(e)}")
        # Continue anyway - RAG will be initialized on first use
    
    yield
    
    # Shutdown
    logger.info("Shutting down Medical AI Assistant")


# FastAPI app
app = FastAPI(
    title="Medical AI Assistant API",
    description="Multi-LLM medical chatbot with RAG capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Medical AI Assistant API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Verifies Redis and ChromaDB connectivity.
    """
    # Check Redis
    redis_connected = False
    try:
        r = redis.Redis.from_url(settings.celery_broker_url)
        r.ping()
        redis_connected = True
    except:
        pass
    
    # Check ChromaDB
    chroma_initialized = False
    try:
        if rag_retriever and rag_retriever._initialized:
            chroma_initialized = True
    except:
        pass
    
    return HealthResponse(
        status="healthy" if redis_connected else "degraded",
        version="1.0.0",
        redis_connected=redis_connected,
        chroma_initialized=chroma_initialized
    )


@app.post("/api/v1/query", tags=["Query"])
async def submit_query(request: QueryRequest):
    """
    Submit medical query for processing.
    
    Returns task_id immediately. Use /api/v1/task/{task_id} to get result.
    """
    try:
        logger.info(f"Received query: model={request.model_choice}, rag={request.use_rag}")
        
        # Submit to Celery
        task = query_task.delay(
            query=request.query,
            model_choice=request.model_choice.value,
            use_rag=request.use_rag,
            api_keys=request.api_keys
        )
        
        logger.info(f"Task submitted: {task.id}")
        
        return {
            "task_id": task.id,
            "status": "PENDING",
            "message": "Query submitted for processing"
        }
        
    except Exception as e:
        logger.error(f"Failed to submit query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/task/{task_id}", response_model=QueryResponse, tags=["Query"])
async def get_task_status(task_id: str):
    """
    Get status and result of submitted task.
    
    Poll this endpoint to get query results.
    """
    try:
        task_result = AsyncResult(task_id, app=celery_app)
        
        status = TaskStatus(task_result.status)
        
        response = QueryResponse(
            task_id=task_id,
            status=status,
            timestamp=datetime.utcnow()
        )
        
        if status == TaskStatus.SUCCESS:
            result = task_result.result
            response.answer = result.get("answer")
            response.model_used = result.get("model_used")
            response.latency_ms = result.get("latency_ms")
            response.rag_used = result.get("rag_used", False)
            response.retrieved_docs = result.get("retrieved_docs")
            
            # Record metrics
            metrics_tracker.record_query(QueryMetrics(
                timestamp=datetime.utcnow(),
                query="[query]",  # Don't store full query for privacy
                model_used=response.model_used,
                latency_ms=response.latency_ms,
                rag_used=response.rag_used,
                success=True
            ))
            
        elif status == TaskStatus.FAILURE:
            response.error = str(task_result.result)
            
            # Record failure
            metrics_tracker.record_query(QueryMetrics(
                timestamp=datetime.utcnow(),
                query="[query]",
                model_used="unknown",
                latency_ms=0,
                rag_used=False,
                success=False,
                error=response.error
            ))
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get task status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/metrics", tags=["Metrics"])
async def get_metrics():
    """Get performance metrics."""
    return metrics_tracker.get_summary()


@app.get("/api/v1/rag/stats", tags=["RAG"])
async def get_rag_stats():
    """Get RAG system statistics."""
    if not rag_retriever:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        return rag_retriever.get_stats()
    except Exception as e:
        logger.error(f"Failed to get RAG stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
