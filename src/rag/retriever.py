import sys
from pathlib import Path
from typing import List, Dict, Optional


from src.storage.vector_store import VectorStore
from src.embedding.embedder import NeuMLEmbedder, BaseEmbedder


class Retriever:    
    def __init__(self, vector_store: Optional[VectorStore] = None, embedder: Optional[BaseEmbedder] = None):
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or NeuMLEmbedder()
    
    def retrieve(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        query_embedding = self.embedder.encode([query])[0].tolist()
        chunks = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
        return chunks


