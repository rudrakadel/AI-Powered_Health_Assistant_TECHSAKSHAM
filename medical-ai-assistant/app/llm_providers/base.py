"""
Abstract base class for LLM providers.
Defines unified interface for Ollama, OpenAI, Gemini.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class LLMResponse:
    """Standardized LLM response format."""
    content: str
    model: str
    latency_ms: float
    token_count: Optional[int] = None
    metadata: Optional[Dict] = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All providers (Ollama, OpenAI, Gemini) implement this interface,
    enabling seamless switching and fallback strategies.
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize provider.
        
        Args:
            api_key: API key for cloud providers (not needed for Ollama)
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text completion.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters
            
        Returns:
            LLMResponse with generated text and metadata
            
        Raises:
            Exception: If generation fails
        """
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """
        Check if provider is available and reachable.
        
        Returns:
            True if provider can be used, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return human-readable provider name."""
        pass
