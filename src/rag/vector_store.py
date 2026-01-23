import sys
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional

root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        db_config = self.config.get('vector_db', {})
        self.persist_directory = db_config.get('persist_directory', 'storage/chroma')
        self.collection_name = db_config.get('collection_name', 'alzheimer_targets')
        
        
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        ids = [chunk['chunk_id'] for chunk in chunks]
        texts = [chunk['text'] for chunk in chunks]
        
        metadatas = []
        for chunk in chunks:
            metadata = {
                'doc_id': chunk['doc_id'],
                'section': chunk['section'],
                'title': chunk['meta'].get('title', ''),
                'year': str(chunk['meta'].get('year', '')),
                'source': chunk['meta'].get('source', ''),
                'pmid': chunk['meta'].get('pmid', ''),
                'url': chunk['meta'].get('url', '')
            }
            metadatas.append(metadata)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    
    def search(self, query_embedding: List[float], top_k: int = 10, 
               filters: Optional[Dict] = None) -> List[Dict]:
        where = filters if filters else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )
        
        chunks = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                chunk = {
                    'chunk_id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                chunks.append(chunk)
        
        return chunks
    
    def load_from_chunks_file(
        self,
        chunks_path: str = "data/chunks.jsonl",
        embedder=None,
        overwrite: bool = False
    ):
        if embedder is None:
            root_dir = Path(__file__).parent.parent.parent
            if str(root_dir) not in sys.path:
                sys.path.insert(0, str(root_dir))
            from src.embedding.embedder import Embedder
            embedder = Embedder()
        
        chunks_path = Path(chunks_path)
        if not chunks_path.exists():
            return
        
        chunks, embeddings = embedder.process_chunks(str(chunks_path))
        
        if chunks is None or embeddings is None:
            return
        
        embeddings_list = embeddings.tolist()
        
        existing_count = self.collection.count()
        if existing_count > 0:
            if not overwrite:
                return
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        self.add_chunks(chunks, embeddings_list)


