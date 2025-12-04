"""
Ollama LLM provider implementation.
"""

import time
import asyncio
from typing import Optional

try:
    import ollama
except ImportError:
    ollama = None

from .base import BaseLLMProvider, LLMResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaLLMProvider(BaseLLMProvider):
    """Ollama provider for local LLM inference."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:latest", **kwargs):
        super().__init__(**kwargs)
        if ollama is None:
            raise ImportError("ollama package not installed. Run: pip install ollama")
        self.base_url = base_url
        self.model = model
        self.client = ollama.Client(host=base_url)
        logger.info(f"Initialized Ollama provider: {base_url}, model: {model}")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Ollama."""
        start_time = time.time()
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens or 2048
                    }
                )
            )
            
            latency_ms = (time.time() - start_time) * 1000
            content = response["message"]["content"]
            
            logger.info(f"Ollama generation successful: {latency_ms:.2f}ms")
            
            return LLMResponse(
                content=content,
                model=f"ollama/{self.model}",
                latency_ms=latency_ms,
                metadata={
                    "eval_count": response.get("eval_count"),
                    "total_duration": response.get("total_duration")
                }
            )
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {str(e)}")
            raise Exception(f"Ollama error: {str(e)}")
    
    async def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(None, self.client.list)
            available_models = [m["name"] for m in models.get("models", [])]
            return self.model in available_models or f"{self.model}:latest" in available_models
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {str(e)}")
            return False
    
    @property
    def provider_name(self) -> str:
        return f"Ollama ({self.model})"
