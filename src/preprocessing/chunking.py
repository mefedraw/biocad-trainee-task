import json
from pathlib import Path
from typing import Dict, List
import tiktoken
import yaml
from transformers import AutoTokenizer


class BaseTokenizerAdapter:
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError


class TikTokenAdapter(BaseTokenizerAdapter):
    def __init__(self, encoding_name: str = "cl100k_base"):
        self._tokenizer = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)


class HuggingFaceTokenizerAdapter(BaseTokenizerAdapter):
    def __init__(self, model_name: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False) #[CLS][SEP] убираем чтобы считать длину правильно


class SectionAwareChunker:
    def __init__(
        self,
        config_path: str = "config.yaml",
        tokenizer_adapter: BaseTokenizerAdapter | None = None,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        chunk_config = self.config.get("chunking", {})
        self.chunk_size = chunk_config.get("chunk_size", 500)
        self.chunk_overlap = chunk_config.get("chunk_overlap", 100)
        self.min_chunk_size = chunk_config.get("min_chunk_size", 100)

        self.tokenizer = tokenizer_adapter or TikTokenAdapter("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def chunk_text(self, text: str, section: str, doc_meta: Dict) -> List[Dict]:
        if not text or self.count_tokens(text.strip()) < self.min_chunk_size:
            return []

        chunks = []
        tokens = self.count_tokens(text)

        if tokens <= self.chunk_size:
            return [self._build_chunk(text.strip(), section, doc_meta, 0)]

        words = text.split()
        current_chunk = []
        current_tokens = 0
        chunk_idx = 0
        i = 0

        while i < len(words):
            word = words[i]
            word_tokens = self.count_tokens(word)

            if current_tokens + word_tokens <= self.chunk_size:
                current_chunk.append(word)
                current_tokens += word_tokens
                i += 1
            else:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(self._build_chunk(chunk_text, section, doc_meta, chunk_idx))
                    chunk_idx += 1

                overlap_tokens = 0
                overlap_words = []
                j = len(current_chunk) - 1

                while j >= 0 and overlap_tokens < self.chunk_overlap:
                    overlap_word = current_chunk[j]
                    overlap_word_tokens = self.count_tokens(overlap_word)

                    if overlap_tokens + overlap_word_tokens <= self.chunk_overlap:
                        overlap_words.insert(0, overlap_word)
                        overlap_tokens += overlap_word_tokens
                        j -= 1
                    else:
                        break

                current_chunk = overlap_words
                current_tokens = overlap_tokens

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if self.count_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append(self._build_chunk(chunk_text, section, doc_meta, chunk_idx))

        return chunks

    def _build_chunk(self, text: str, section: str, doc_meta: Dict, chunk_idx: int) -> Dict:
        return {
            "chunk_id": f"{doc_meta['doc_id']}_{section}_{chunk_idx:02d}",
            "doc_id": doc_meta["doc_id"],
            "section": section,
            "text": text,
            "meta": {
                "title": doc_meta.get("title", ""),
                "year": doc_meta.get("year"),
                "source": doc_meta.get("source", ""),
                "pmid": doc_meta.get("pmid", ""),
                "doi": doc_meta.get("doi", ""),
                "url": doc_meta.get("url", ""),
            },
        }

    def process_documents(
        self,
        input_path: str = "data/clean_docs.jsonl",
        output_path: str = "data/chunks.jsonl",
    ):
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            return

        all_chunks = []

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                sections = doc.get("sections", {})
                for section_name, section_text in sections.items():
                    if section_text and len(section_text.strip()) > 0:
                        chunks = self.chunk_text(section_text, section_name, doc)
                        if chunks:
                            all_chunks.extend(chunks)

        if not all_chunks:
            return

        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")