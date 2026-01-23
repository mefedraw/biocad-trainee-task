import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, chunks: List[Dict], top_k: int = 8) -> List[Dict]:
        if not self.model or not chunks:
            return chunks[:top_k]
        
        pairs = [[query, chunk['text']] for chunk in chunks]
        scores = self.model.predict(pairs)
        
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        reranked = []
        for chunk, score in scored_chunks[:top_k]:
            if 'metadata' not in chunk:
                chunk['metadata'] = {}
            chunk['metadata']['rerank_score'] = float(score)
            reranked.append(chunk)
        
        return reranked
