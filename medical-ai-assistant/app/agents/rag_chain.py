"""
LangChain-based RAG implementation.
Combines retrieval with LLM generation.
"""

from typing import List, Optional
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from app.llm_providers.base import BaseLLMProvider, LLMResponse
from app.rag.retriever import RAGRetriever
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGChain:
    """
    RAG chain that combines document retrieval with LLM generation.
    
    Workflow:
    1. Retrieve relevant documents from vector store
    2. Format documents as context
    3. Build prompt with system instructions + context + query
    4. Generate answer using selected LLM
    5. Return structured response
    """
    
    # System prompt for medical assistant
    SYSTEM_PROMPT = """You are an expert medical AI assistant with access to a curated medical knowledge base.

Your role:
- Provide accurate, evidence-based medical information
- Use the retrieved medical knowledge as your primary source
- If the knowledge base doesn't contain relevant information, clearly state this
- Never make up medical information
- Always include appropriate disclaimers
- Be clear, concise, and empathetic

Important disclaimers:
- This is informational only, not medical advice
- Users should consult qualified healthcare professionals
- Emergency situations require immediate medical attention

Retrieved medical knowledge is provided below. Use it to answer the user's question."""
    
    def __init__(self, retriever: RAGRetriever):
        self.retriever = retriever
        logger.info("RAGChain initialized")
    
    async def generate_answer(
        self,
        query: str,
        llm_provider: BaseLLMProvider,
        use_rag: bool = True,
        top_k: int = 5
    ) -> dict:
        """
        Generate answer to medical query.
        
        Args:
            query: User question
            llm_provider: LLM provider to use
            use_rag: Whether to retrieve and use context
            top_k: Number of documents to retrieve
            
        Returns:
            Dict with answer, metadata, and retrieved docs
        """
        try:
            retrieved_docs = []
            context = ""
            
            # Step 1: Retrieve relevant documents if RAG enabled
            if use_rag:
                logger.info(f"Retrieving {top_k} relevant documents")
                retrieved_docs = self.retriever.retrieve_relevant_docs(query, k=top_k)
                context = self.retriever.format_context(retrieved_docs)
                logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            # Step 2: Build prompt
            if use_rag and context:
                full_prompt = f"""{context}

User Question: {query}

Instructions:
- Answer based primarily on the retrieved medical knowledge above
- If information is insufficient, clearly state this
- Include the standard medical disclaimer
- Be concise but thorough

Answer:"""
            else:
                full_prompt = f"""User Question: {query}

Instructions:
- Provide a helpful medical answer
- Include appropriate disclaimers
- Be concise but thorough

Answer:"""
            
            # Step 3: Generate answer using LLM
            logger.info(f"Generating answer using {llm_provider.provider_name}")
            llm_response: LLMResponse = await llm_provider.generate(
                prompt=full_prompt,
                system_prompt=self.SYSTEM_PROMPT if not use_rag else None,  # System prompt included in full_prompt if RAG
                temperature=0.3,  # Lower temperature for medical accuracy
                max_tokens=1024
            )
            
            # Step 4: Format response
            result = {
                "answer": llm_response.content,
                "model_used": llm_response.model,
                "latency_ms": llm_response.latency_ms,
                "rag_used": use_rag,
                "retrieved_docs": [
                    {
                        "question": doc.metadata.get("question", ""),
                        "answer": doc.metadata.get("answer", ""),
                        "tags": doc.metadata.get("tags", [])
                    }
                    for doc in retrieved_docs
                ] if retrieved_docs else [],
                "token_count": llm_response.token_count,
                "metadata": llm_response.metadata
            }
            
            logger.info("Answer generation complete")
            return result
            
        except Exception as e:
            logger.error(f"RAG chain failed: {str(e)}")
            raise
    
    def add_disclaimer(self, answer: str) -> str:
        """Add medical disclaimer to answer."""
        disclaimer = "\n\n⚠️ **Medical Disclaimer**: This information is for educational purposes only and does not constitute medical advice. Please consult a qualified healthcare professional for diagnosis and treatment."
        
        if disclaimer.lower() not in answer.lower():
            return answer + disclaimer
        return answer
