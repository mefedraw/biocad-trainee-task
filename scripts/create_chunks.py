import sys
from pathlib import Path
from src.preprocessing.chunking import SectionAwareChunker


def main():
    input_path = Path("data/clean_docs.jsonl")
    if not input_path.exists():
        return
    
    chunker = SectionAwareChunker(
    tokenizer_adapter=HuggingFaceTokenizerAdapter(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        )
    )
    
    chunker.process_documents()


if __name__ == "__main__":
    main()
