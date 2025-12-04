"""
ChromaDB vector store manager.
Handles embedding generation and vector storage.
"""

import chromadb
from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """Manages ChromaDB vector store for medical knowledge base."""
    
    def __init__(self, persist_dir: str = None, collection_name: str = "medical_kb"):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name
        self.embeddings = None
        self.vector_store = None
        logger.info(f"Initialized VectorStore: {self.persist_dir}/{collection_name}")
    
    def initialize(self):
        """Initialize embeddings and vector store."""
        try:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            
            # Use LangChain's HuggingFaceEmbeddings (more stable)
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize ChromaDB
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir
            )
            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    def index_documents(self, documents: List[Dict], batch_size: int = 100):
        """Index documents into ChromaDB."""
        if not self.vector_store:
            self.initialize()
        
        try:
            langchain_docs = [
                Document(page_content=doc["content"], metadata=doc["metadata"])
                for doc in documents
            ]
            
            logger.info(f"Indexing {len(langchain_docs)} documents in batches of {batch_size}")
            
            for i in range(0, len(langchain_docs), batch_size):
                batch = langchain_docs[i:i+batch_size]
                self.vector_store.add_documents(batch)
                logger.info(f"Indexed batch {i//batch_size + 1}/{(len(langchain_docs)-1)//batch_size + 1}")
            
            logger.info("Document indexing complete")
            
        except Exception as e:
            logger.error(f"Failed to index documents: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Search for similar documents."""
        if not self.vector_store:
            self.initialize()
        
        k = k or settings.top_k_results
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(results)} documents for query")
            return results
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            return []
    
    def is_empty(self) -> bool:
        """Check if vector store is empty."""
        if not self.vector_store:
            self.initialize()
        
        try:
            collection = self.vector_store._collection
            count = collection.count()
            return count == 0
        except:
            return True
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector store."""
        if not self.vector_store:
            self.initialize()
        
        try:
            collection = self.vector_store._collection
            return {
                "name": self.collection_name,
                "count": collection.count(),
                "embedding_model": settings.embedding_model
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {}
