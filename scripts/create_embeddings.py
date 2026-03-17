import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding.embedder import NeuMLEmbedder
from src.rag.vector_store import VectorStore


def main():
    input_path = Path("data/chunks.jsonl")
    if not input_path.exists():
        return
    
    embedder = NeuMLEmbedder()
    store = VectorStore()
    store.load_from_chunks_file(embedder=embedder)


if __name__ == "__main__":
    main()
