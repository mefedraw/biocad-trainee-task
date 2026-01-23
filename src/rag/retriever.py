import sys
from pathlib import Path
from typing import List, Dict, Optional

root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.rag.vector_store import VectorStore
from src.embedding.embedder import Embedder


class Retriever:    
    def __init__(self, vector_store: Optional[VectorStore] = None, embedder: Optional[Embedder] = None):
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or Embedder()
    
    def retrieve(self, query: str, top_k: int = 30, filters: Optional[Dict] = None) -> List[Dict]:
        query_embedding = self.embedder.encode([query])[0].tolist()
        chunks = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
        return chunks


