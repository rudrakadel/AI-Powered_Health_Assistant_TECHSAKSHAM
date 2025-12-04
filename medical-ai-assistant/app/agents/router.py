"""
Query router for intelligent model selection.
Decides which LLM to use based on query characteristics.
"""

from typing import Dict
from app.models.schemas import ModelChoice
from app.utils.logger import get_logger

logger = get_logger(__name__)


class QueryRouter:
    """
    Routes queries to appropriate LLM based on complexity and availability.
    
    Routing strategy:
    - Simple FAQ → Ollama (fast, local)
    - Complex reasoning → OpenAI/Gemini (more capable)
    - Auto mode → Intelligent selection based on query analysis
    """
    
    def __init__(self):
        self.simple_keywords = [
            "what is", "define", "meaning of", "explain",
            "symptoms", "causes", "treatment for"
        ]
        self.complex_keywords = [
            "compare", "analyze", "why", "how does", "relationship",
            "differential diagnosis", "complicated by"
        ]
        logger.info("QueryRouter initialized")
    
    def route(self, query: str, model_choice: ModelChoice, available_providers: Dict[str, bool]) -> str:
        """
        Determine which LLM to use.
        
        Args:
            query: User question
            model_choice: User's model preference
            available_providers: Dict of provider availability (e.g., {"ollama": True, "openai": False})
            
        Returns:
            Provider name to use ("ollama", "openai", or "gemini")
        """
        query_lower = query.lower()
        
        # If user specified a model, respect it (if available)
        if model_choice != ModelChoice.AUTO:
            provider = model_choice.value
            if available_providers.get(provider, False):
                logger.info(f"Using user-specified provider: {provider}")
                return provider
            else:
                logger.warning(f"User requested {provider} but not available, falling back to auto")
        
        # Auto mode: intelligent routing
        logger.info("Auto-routing based on query complexity")
        
        # Check if simple query
        is_simple = any(kw in query_lower for kw in self.simple_keywords)
        is_complex = any(kw in query_lower for kw in self.complex_keywords)
        
        # Routing logic with fallback chain
        if is_simple and not is_complex:
            # Simple query: prefer Ollama (fast, local)
            if available_providers.get("ollama", False):
                logger.info("Simple query → Ollama")
                return "ollama"
        
        # Complex query or Ollama unavailable: prefer cloud providers
        if available_providers.get("openai", False):
            logger.info("Complex query → OpenAI")
            return "openai"
        
        if available_providers.get("gemini", False):
            logger.info("Complex query → Gemini")
            return "gemini"
        
        # Fallback to Ollama if nothing else available
        if available_providers.get("ollama", False):
            logger.info("Fallback → Ollama")
            return "ollama"
        
        # No providers available
        logger.error("No LLM providers available!")
        raise Exception("No LLM providers are currently available")
    
    def estimate_complexity(self, query: str) -> Dict[str, float]:
        """
        Estimate query complexity.
        
        Returns:
            Dict with complexity scores
        """
        query_lower = query.lower()
        
        simple_score = sum(1 for kw in self.simple_keywords if kw in query_lower) / len(self.simple_keywords)
        complex_score = sum(1 for kw in self.complex_keywords if kw in query_lower) / len(self.complex_keywords)
        
        return {
            "simple_score": simple_score,
            "complex_score": complex_score,
            "query_length": len(query),
            "word_count": len(query.split())
        }
