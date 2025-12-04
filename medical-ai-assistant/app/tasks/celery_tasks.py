"""
Celery tasks for async processing.
Handles long-running LLM and RAG operations.
"""

import asyncio
from celery import Celery
from typing import Dict
from app.config import settings
from app.models.schemas import ModelChoice
from app.llm_providers import OllamaLLMProvider, OpenAILLMProvider, GeminiLLMProvider
from app.rag.retriever import RAGRetriever
from app.agents.router import QueryRouter
from app.agents.rag_chain import RAGChain
from app.utils.logger import get_logger
from tenacity import retry, stop_after_attempt, wait_exponential

logger = get_logger(__name__)

# Initialize Celery
celery_app = Celery(
    "medical_ai_assistant",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minute timeout
    task_soft_time_limit=270  # 4.5 minute soft timeout
)

# Global RAG components (initialized once per worker)
rag_retriever = None
query_router = None


def get_rag_retriever() -> RAGRetriever:
    """Lazy initialization of RAG retriever."""
    global rag_retriever
    if rag_retriever is None:
        logger.info("Initializing RAG retriever in Celery worker")
        rag_retriever = RAGRetriever()
        rag_retriever.initialize()
    return rag_retriever


def get_query_router() -> QueryRouter:
    """Lazy initialization of query router."""
    global query_router
    if query_router is None:
        logger.info("Initializing query router in Celery worker")
        query_router = QueryRouter()
    return query_router


def get_llm_provider(provider_name: str, api_keys: Dict[str, str] = None):
    """
    Create LLM provider instance.
    
    Args:
        provider_name: "ollama", "openai", or "gemini"
        api_keys: Optional API keys from GUI
        
    Returns:
        Initialized LLM provider
    """
    api_keys = api_keys or {}
    
    if provider_name == "ollama":
        return OllamaLLMProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model
        )
    elif provider_name == "openai":
        api_key = api_keys.get("openai") or settings.openai_api_key
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        return OpenAILLMProvider(api_key=api_key, model="gpt-4")
    elif provider_name == "gemini":
        api_key = api_keys.get("gemini") or settings.gemini_api_key
        if not api_key:
            raise ValueError("Gemini API key not provided")
        return GeminiLLMProvider(api_key=api_key, model="gemini-1.5-pro")
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


async def check_provider_availability(provider_name: str, api_keys: Dict[str, str] = None) -> bool:
    """Check if LLM provider is available."""
    try:
        provider = get_llm_provider(provider_name, api_keys)
        return await provider.is_available()
    except Exception as e:
        logger.warning(f"Provider {provider_name} not available: {str(e)}")
        return False


@celery_app.task(bind=True, name="query_task")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def query_task(self, query: str, model_choice: str, use_rag: bool, api_keys: Dict[str, str] = None):
    """
    Main async task for processing medical queries.
    
    Workflow:
    1. Initialize RAG and router
    2. Check LLM provider availability
    3. Route to appropriate LLM
    4. Generate answer with RAG (if enabled)
    5. Handle errors with fallback chain
    
    Args:
        query: User question
        model_choice: "ollama", "openai", "gemini", or "auto"
        use_rag: Whether to use RAG
        api_keys: Optional API keys from GUI
        
    Returns:
        Dict with answer and metadata
    """
    logger.info(f"Processing query task: model={model_choice}, rag={use_rag}")
    
    try:
        # Update task state
        self.update_state(state='STARTED', meta={'status': 'Initializing...'})
        
        # Initialize components
        retriever = get_rag_retriever()
        router = get_query_router()
        rag_chain = RAGChain(retriever)
        
        # Check provider availability
        self.update_state(state='STARTED', meta={'status': 'Checking LLM availability...'})
        
        available_providers = {}
        for provider in ["ollama", "openai", "gemini"]:
            available_providers[provider] = asyncio.run(
                check_provider_availability(provider, api_keys)
            )
        
        logger.info(f"Available providers: {available_providers}")
        
        # Route to appropriate provider
        model_choice_enum = ModelChoice(model_choice)
        selected_provider = router.route(query, model_choice_enum, available_providers)
        
        logger.info(f"Selected provider: {selected_provider}")
        
        # Generate answer
        self.update_state(state='STARTED', meta={'status': f'Generating answer with {selected_provider}...'})
        
        provider = get_llm_provider(selected_provider, api_keys)
        
        result = asyncio.run(
            rag_chain.generate_answer(
                query=query,
                llm_provider=provider,
                use_rag=use_rag,
                top_k=settings.top_k_results
            )
        )
        
        # Add disclaimer
        result["answer"] = rag_chain.add_disclaimer(result["answer"])
        
        logger.info(f"Query task completed successfully: {result['latency_ms']:.2f}ms")
        
        return result
        
    except Exception as e:
        logger.error(f"Query task failed: {str(e)}")
        
        # Try fallback providers
        fallback_order = ["ollama", "openai", "gemini"]
        fallback_order = [p for p in fallback_order if p != selected_provider]
        
        for fallback_provider in fallback_order:
            if available_providers.get(fallback_provider, False):
                logger.info(f"Trying fallback provider: {fallback_provider}")
                try:
                    provider = get_llm_provider(fallback_provider, api_keys)
                    result = asyncio.run(
                        rag_chain.generate_answer(
                            query=query,
                            llm_provider=provider,
                            use_rag=use_rag,
                            top_k=settings.top_k_results
                        )
                    )
                    result["answer"] = rag_chain.add_disclaimer(result["answer"])
                    logger.info(f"Fallback successful with {fallback_provider}")
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback {fallback_provider} failed: {str(fallback_error)}")
                    continue
        
        # All providers failed
        raise Exception(f"All LLM providers failed. Original error: {str(e)}")
