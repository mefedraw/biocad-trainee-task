import sys
import json
import yaml
from pathlib import Path
from typing import List, Dict
import tiktoken

root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


class SectionAwareChunker:
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        chunk_config = self.config.get('chunking', {})
        self.chunk_size = chunk_config.get('chunk_size', 500)
        self.chunk_overlap = chunk_config.get('chunk_overlap', 100)
        self.min_chunk_size = chunk_config.get('min_chunk_size', 100)
        
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return len(text) // 4
    
    def chunk_text(self, text: str, section: str, doc_meta: Dict) -> List[Dict]:
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        chunks = []
        tokens = self.count_tokens(text)
        
        if tokens <= self.chunk_size:
            chunk = {
                "chunk_id": f"{doc_meta['doc_id']}_{section}_00",
                "doc_id": doc_meta['doc_id'],
                "section": section,
                "text": text.strip(),
                "meta": {
                    "title": doc_meta.get('title', ''),
                    "year": doc_meta.get('year'),
                    "source": doc_meta.get('source', ''),
                    "pmid": doc_meta.get('pmid', ''),
                    "doi": doc_meta.get('doi', ''),
                    "url": doc_meta.get('url', '')
                }
            }
            chunks.append(chunk)
            return chunks
        
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
                    chunk_text = ' '.join(current_chunk)
                    chunk = {
                        "chunk_id": f"{doc_meta['doc_id']}_{section}_{chunk_idx:02d}",
                        "doc_id": doc_meta['doc_id'],
                        "section": section,
                        "text": chunk_text,
                        "meta": {
                            "title": doc_meta.get('title', ''),
                            "year": doc_meta.get('year'),
                            "source": doc_meta.get('source', ''),
                            "pmid": doc_meta.get('pmid', ''),
                            "doi": doc_meta.get('doi', ''),
                            "url": doc_meta.get('url', '')
                        }
                    }
                    chunks.append(chunk)
                    chunk_idx += 1
                
                overlap_tokens = 0
                overlap_words = []
                j = len(current_chunk) - 1
                
                while j >= 0 and overlap_tokens < self.chunk_overlap:
                    word = current_chunk[j]
                    word_tokens = self.count_tokens(word)
                    if overlap_tokens + word_tokens <= self.chunk_overlap:
                        overlap_words.insert(0, word)
                        overlap_tokens += word_tokens
                        j -= 1
                    else:
                        break
                
                current_chunk = overlap_words
                current_tokens = overlap_tokens
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if self.count_tokens(chunk_text) >= self.min_chunk_size:
                chunk = {
                    "chunk_id": f"{doc_meta['doc_id']}_{section}_{chunk_idx:02d}",
                    "doc_id": doc_meta['doc_id'],
                    "section": section,
                    "text": chunk_text,
                    "meta": {
                        "title": doc_meta.get('title', ''),
                        "year": doc_meta.get('year'),
                        "source": doc_meta.get('source', ''),
                        "pmid": doc_meta.get('pmid', ''),
                        "doi": doc_meta.get('doi', ''),
                        "url": doc_meta.get('url', '')
                    }
                }
                chunks.append(chunk)
        
        return chunks
    
    def process_documents(self, input_path: str = "data/clean_docs.jsonl", 
                         output_path: str = "data/chunks.jsonl"):
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not input_path.exists():
            return
        
        all_chunks = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                sections = doc.get('sections', {})
                for section_name, section_text in sections.items():
                    if section_text and len(section_text.strip()) > 0:
                        chunks = self.chunk_text(section_text, section_name, doc)
                        if chunks:
                            all_chunks.extend(chunks)
        
        if not all_chunks:
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')


