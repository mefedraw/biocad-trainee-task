import sys
import json
from pathlib import Path
from src.embedding.embedder import Embedder
from src.storage.vector_store import VectorStore


def main():
    chunks_path = Path("data/chunks.jsonl")
    if not chunks_path.exists():
        print(f"chunks file not found: {chunks_path}")
        return

    # читаем чанки
    chunks = []
    texts = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk.get("text", "").strip():
                chunks.append(chunk)
                texts.append(chunk["text"])

    if not chunks:
        print("no chunks to embed")
        return

    print(f"encoding {len(chunks)} chunks...")
    embedder = Embedder()
    embeddings = embedder.encode(texts)

    store = VectorStore()

    # пересоздаём коллекцию
    existing = [c.name for c in store.client.get_collections().collections]
    if store.collection_name in existing:
        store.client.delete_collection(store.collection_name)

    store.add_chunks(chunks, embeddings.tolist())
    print(f"loaded {len(chunks)} chunks into qdrant")


if __name__ == "__main__":
    main()
