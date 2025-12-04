"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, List
from enum import Enum
from datetime import datetime


class ModelChoice(str, Enum):
    """Available LLM model options."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"
    AUTO = "auto"


class QueryRequest(BaseModel):
    """Request schema for medical query endpoint."""
    query: str = Field(..., min_length=1, max_length=2000, description="Medical question")
    model_choice: ModelChoice = Field(default=ModelChoice.AUTO, description="LLM to use")
    use_rag: bool = Field(default=True, description="Enable RAG retrieval")
    api_keys: Optional[Dict[str, str]] = Field(default=None, description="API keys override")
    
    model_config = ConfigDict(
        protected_namespaces=(),  # Allow model_* field names
        json_schema_extra={
            "example": {
                "query": "What are the symptoms of diabetes?",
                "model_choice": "auto",
                "use_rag": True,
                "api_keys": {
                    "openai": "sk-...",
                    "gemini": "AIza..."
                }
            }
        }
    )


class TaskStatus(str, Enum):
    """Celery task status enum."""
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"


class QueryResponse(BaseModel):
    """Response schema for query results."""
    task_id: str = Field(..., description="Celery task ID")
    status: TaskStatus = Field(..., description="Task status")
    answer: Optional[str] = Field(None, description="Generated answer")
    model_used: Optional[str] = Field(None, description="Which LLM was used")
    latency_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    rag_used: bool = Field(default=False, description="Whether RAG was used")
    retrieved_docs: Optional[List[Dict]] = Field(None, description="Retrieved documents metadata")
    error: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(
        protected_namespaces=(),  # Allow model_* field names
        json_schema_extra={
            "example": {
                "task_id": "abc123",
                "status": "SUCCESS",
                "answer": "Diabetes symptoms include...",
                "model_used": "gpt-4",
                "latency_ms": 1250.5,
                "rag_used": True,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    redis_connected: bool
    chroma_initialized: bool
