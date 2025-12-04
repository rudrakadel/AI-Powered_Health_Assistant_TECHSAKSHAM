"""
Medical chatbot dataset loader.
Loads and preprocesses the Kaggle medical dataset.
"""

import pandas as pd
import ast
from typing import List, Dict
from pathlib import Path
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MedicalDatasetLoader:
    """
    Loads and processes the medical chatbot dataset.
    
    Dataset schema:
    - short_question: User question
    - short_answer: Expert answer
    - tags: List of medical tags (as string representation)
    - label: Quality indicator (1=good, -1=bad)
    """
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or settings.dataset_path
        self.df = None
        logger.info(f"Initialized dataset loader: {self.dataset_path}")
    
    def load(self, filter_by_label: bool = True) -> pd.DataFrame:
        """
        Load dataset from CSV.
        
        Args:
            filter_by_label: If True, only keep rows with label=1 (high quality)
            
        Returns:
            Loaded DataFrame
        """
        try:
            logger.info(f"Loading dataset from: {self.dataset_path}")
            
            # Verify file exists
            if not Path(self.dataset_path).exists():
                raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            
            # Load CSV
            self.df = pd.read_csv(self.dataset_path)
            
            logger.info(f"Loaded {len(self.df)} rows")
            
            # Filter by label if requested
            if filter_by_label and 'label' in self.df.columns:
                original_count = len(self.df)
                self.df = self.df[self.df['label'] == 1]
                logger.info(f"Filtered to {len(self.df)} high-quality rows (removed {original_count - len(self.df)})")
            
            # Parse tags column (convert string representation to list)
            if 'tags' in self.df.columns:
                self.df['tags_parsed'] = self.df['tags'].apply(self._parse_tags)
            
            return self.df
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
    
    def _parse_tags(self, tags_str: str) -> List[str]:
        """Parse tags from string representation to list."""
        try:
            if pd.isna(tags_str) or not tags_str:
                return []
            # Handle string representation of list
            return ast.literal_eval(tags_str)
        except:
            return []
    
    def get_documents(self) -> List[Dict]:
        """
        Convert dataset rows to documents for RAG indexing.
        
        Each document combines question, answer, and tags into
        searchable text with metadata.
        
        Returns:
            List of document dictionaries
        """
        if self.df is None:
            self.load()
        
        documents = []
        
        for idx, row in self.df.iterrows():
            # Combine fields into searchable content
            content_parts = []
            
            if pd.notna(row.get('short_question')):
                content_parts.append(f"Question: {row['short_question']}")
            
            if pd.notna(row.get('short_answer')):
                content_parts.append(f"Answer: {row['short_answer']}")
            
            # Parse tags
            tags = row.get('tags_parsed', [])
            if tags:
                content_parts.append(f"Tags: {', '.join(tags)}")
            
            content = "\n".join(content_parts)
            
            # ✅ FIX: Convert list metadata to strings for ChromaDB compatibility
            tags_str = ', '.join(str(t) for t in tags) if tags else ''
            
            # Create document with metadata (ChromaDB only accepts str/int/float/bool)
            doc = {
                "content": content,
                "metadata": {
                    "question": str(row.get('short_question', '')),
                    "answer": str(row.get('short_answer', '')),
                    "tags": tags_str,  # ← Convert list to comma-separated string
                    "label": int(row.get('label', 1)),
                    "row_id": int(idx)
                }
            }
            
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} documents for indexing")
        return documents
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        if self.df is None:
            self.load()
        
        stats = {
            "total_rows": len(self.df),
            "columns": list(self.df.columns),
            "avg_question_length": self.df['short_question'].str.len().mean() if 'short_question' in self.df else 0,
            "avg_answer_length": self.df['short_answer'].str.len().mean() if 'short_answer' in self.df else 0,
            "unique_tags": len(set([tag for tags in self.df.get('tags_parsed', []) if tags for tag in tags]))
        }
        
        return stats
