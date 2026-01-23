import yaml
from pathlib import Path
from typing import Dict, Optional


_ARTICLES_MAPPING = None


def _load_articles_mapping() -> Dict:
    global _ARTICLES_MAPPING
    
    if _ARTICLES_MAPPING is not None:
        return _ARTICLES_MAPPING
    
    root_dir = Path(__file__).parent
    mapping_path = root_dir / "articles_mapping.yaml"
    
    if not mapping_path.exists():
        _ARTICLES_MAPPING = {}
        return _ARTICLES_MAPPING
    
    with open(mapping_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        _ARTICLES_MAPPING = data.get('articles', {})
    return _ARTICLES_MAPPING


def get_article_info(doc_id: str) -> Optional[Dict]:
    if not doc_id:
        return None
    
    articles_mapping = _load_articles_mapping()
    
    if doc_id in articles_mapping:
        return articles_mapping[doc_id]
    
    doc_id_clean = doc_id.replace('.full', '').replace('.pdf', '').replace('-main', '').strip()
    
    if doc_id_clean in articles_mapping:
        return articles_mapping[doc_id_clean]
    
    for key, value in articles_mapping.items():
        key_clean = key.replace('.full', '').replace('.pdf', '').replace('-main', '').strip()
        if key_clean in doc_id_clean or doc_id_clean in key_clean:
            return value
    
    return None
