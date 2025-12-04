"""
Google Gemini LLM provider implementation.
Uses google-generativeai SDK.
"""

import time
from typing import Optional
import google.generativeai as genai
from .base import BaseLLMProvider, LLMResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiLLMProvider(BaseLLMProvider):
    """
    Google Gemini provider.
    
    Supports gemini-1.5-pro, gemini-1.5-flash, etc.
    Requires API key.
    """
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro", **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        if not api_key or api_key == "your-gemini-key-here":
            raise ValueError("Valid Gemini API key required")
        self.model_name = model
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        logger.info(f"Initialized Gemini provider: model={model}")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Gemini API."""
        start_time = time.time()
        
        try:
            # Combine system prompt with user prompt if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser Query: {prompt}"
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens or 2048
            )
            
            response = await self.model.generate_content_async(
                full_prompt,
                generation_config=generation_config
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            content = response.text
            
            # Extract token count if available
            token_count = None
            if hasattr(response, 'usage_metadata'):
                token_count = (
                    response.usage_metadata.prompt_token_count +
                    response.usage_metadata.candidates_token_count
                )
            
            logger.info(f"Gemini generation successful: {latency_ms:.2f}ms, tokens={token_count}")
            
            return LLMResponse(
                content=content,
                model=f"gemini/{self.model_name}",
                latency_ms=latency_ms,
                token_count=token_count,
                metadata={
                    "finish_reason": response.candidates[0].finish_reason if response.candidates else None,
                    "safety_ratings": [
                        {"category": r.category.name, "probability": r.probability.name}
                        for r in response.candidates[0].safety_ratings
                    ] if response.candidates and response.candidates[0].safety_ratings else []
                }
            )
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {str(e)}")
            raise Exception(f"Gemini error: {str(e)}")
    
    async def is_available(self) -> bool:
        """Check if Gemini API is accessible."""
        try:
            # Try listing models to verify connection
            models = genai.list_models()
            return any(self.model_name in m.name for m in models)
        except Exception as e:
            logger.warning(f"Gemini availability check failed: {str(e)}")
            return False
    
    @property
    def provider_name(self) -> str:
        return f"Gemini ({self.model_name})"
