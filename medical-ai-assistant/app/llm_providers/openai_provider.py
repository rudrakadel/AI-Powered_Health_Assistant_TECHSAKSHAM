"""
OpenAI LLM provider implementation.
Uses OpenAI Python SDK for GPT models.
"""

import time
from typing import Optional
from openai import AsyncOpenAI
from .base import BaseLLMProvider, LLMResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAILLMProvider(BaseLLMProvider):
    """
    OpenAI provider for GPT models.
    
    Supports GPT-4, GPT-4o, GPT-3.5-turbo, etc.
    Requires API key.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4", **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        if not api_key or api_key == "your-openai-key-here":
            raise ValueError("Valid OpenAI API key required")
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI provider: model={model}")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        start_time = time.time()
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 2048,
                **kwargs
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            content = response.choices[0].message.content
            token_count = response.usage.total_tokens if response.usage else None
            
            logger.info(f"OpenAI generation successful: {latency_ms:.2f}ms, tokens={token_count}")
            
            return LLMResponse(
                content=content,
                model=f"openai/{self.model}",
                latency_ms=latency_ms,
                token_count=token_count,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
            raise Exception(f"OpenAI error: {str(e)}")
    
    async def is_available(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            # Try a minimal test request
            await self.client.models.retrieve(self.model)
            return True
        except Exception as e:
            logger.warning(f"OpenAI availability check failed: {str(e)}")
            return False
    
    @property
    def provider_name(self) -> str:
        return f"OpenAI ({self.model})"
