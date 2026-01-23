import sys
import json
import yaml
from pathlib import Path
from typing import List, Dict
import numpy as np

root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from sentence_transformers import SentenceTransformer
import torch

class Embedder:
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        embedding_config = self.config.get('embedding', {})
        model_name = embedding_config.get('model', 
            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        device = embedding_config.get('device', 'cpu')
        
        if 'PubMedBERT' in model_name or 'BioBERT' in model_name:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.use_transformers = True
        else:
            self.model = SentenceTransformer(model_name, device=device)
            self.use_transformers = False
            
        self.device = device
        if self.use_transformers:
            self.model = self.model.to(device)
                
    
    def encode(self, texts: List[str]) -> np.ndarray:
        if self.use_transformers:
            embeddings = []
            batch_size = 8
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                    
                    embeddings.append(batch_embeddings)
            
            return np.vstack(embeddings)
        else:
            return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    
    def process_chunks(self, chunks_path: str = "data/chunks.jsonl"):
        chunks_path = Path(chunks_path)
        
        if not chunks_path.exists():
            return None, None
        
        chunks = []
        texts = []
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                if chunk.get('text') and len(chunk['text'].strip()) > 0:
                    chunks.append(chunk)
                    texts.append(chunk['text'])
        
        if not texts:
            return None, None
        
        embeddings = self.encode(texts)
        return chunks, embeddings


