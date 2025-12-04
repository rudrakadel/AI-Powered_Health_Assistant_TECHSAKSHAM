"""
RAG retriever combining vector search with LangChain.
"""

from typing import List, Dict, Optional
from langchain.schema import Document
from app.rag.dataset_loader import MedicalDatasetLoader
from app.rag.vector_store import VectorStoreManager
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGRetriever:
    """
    High-level RAG retrieval interface.
    
    Manages dataset loading, indexing, and retrieval operations.
    Provides simple API for getting relevant medical context.
    """
    
    def __init__(self):
        self.dataset_loader = MedicalDatasetLoader()
        self.vector_store = VectorStoreManager()
        self._initialized = False
        logger.info("RAGRetriever initialized")
    
    def initialize(self, force_reindex: bool = False):
        """
        Initialize RAG system.
        
        Args:
            force_reindex: If True, reload and reindex dataset even if exists
        """
        try:
            logger.info("Initializing RAG system...")
            
            # Initialize vector store
            self.vector_store.initialize()
            
            # Check if already indexed
            if not force_reindex and not self.vector_store.is_empty():
                logger.info("Vector store already contains data, skipping indexing")
                self._initialized = True
                return
            
            # Load dataset
            logger.info("Loading medical dataset...")
            self.dataset_loader.load(filter_by_label=True)
            
            # Get stats
            stats = self.dataset_loader.get_stats()
            logger.info(f"Dataset stats: {stats}")
            
            # Convert to documents and index
            logger.info("Converting dataset to documents...")
            documents = self.dataset_loader.get_documents()
            
            logger.info("Indexing documents into ChromaDB...")
            self.vector_store.index_documents(documents)
            
            self._initialized = True
            logger.info("RAG system initialization complete")
            
        except Exception as e:
            logger.error(f"RAG initialization failed: {str(e)}")
            raise
    
    def retrieve_relevant_docs(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User question
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with metadata
        """
        if not self._initialized:
            logger.warning("RAG not initialized, initializing now...")
            self.initialize()
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Document retrieval failed: {str(e)}")
            return []
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string for LLM.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant medical information found in knowledge base."
        
        context_parts = ["Retrieved Medical Knowledge:\n"]
        
        for i, doc in enumerate(documents, 1):
            # Extract metadata
            metadata = doc.metadata
            question = metadata.get('question', '')
            answer = metadata.get('answer', '')
            tags = metadata.get('tags', [])
            
            # Format as Q&A pair
            context_parts.append(f"\n[Source {i}]")
            if question:
                context_parts.append(f"Q: {question}")
            if answer:
                context_parts.append(f"A: {answer}")
            if tags:
                context_parts.append(f"Tags: {', '.join(tags)}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict:
        """Get RAG system statistics."""
        stats = {
            "initialized": self._initialized,
            "vector_store": self.vector_store.get_collection_stats() if self._initialized else {}
        }
        
        if self.dataset_loader.df is not None:
            stats["dataset"] = self.dataset_loader.get_stats()
        
        return stats
