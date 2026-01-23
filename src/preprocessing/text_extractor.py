import sys
import json
import re
from pathlib import Path
from typing import Dict, Optional, List
import yaml
from bs4 import BeautifulSoup

root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


from grobid_client_python.grobid_client import GrobidClient
import fitz



class TextExtractor:    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.grobid_config = self.config.get('grobid', {})
        self.papers_dir = Path(self.config['paths']['papers_dir'])
        self.output_path = Path(self.config['paths']['processed_data']) / "clean_docs.jsonl"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.grobid_client = GrobidClient(
            grobid_server=self.grobid_config.get('url', 'http://localhost:8070')
        )
    
    def extract_with_grobid(self, pdf_path: Path) -> Optional[Dict]:
        if not self.grobid_client:
            return None
        
        result = self.grobid_client.process_pdf(
            str(pdf_path),
            "processFulltextDocument",
            generate_ids=True,
            consolidate_citations=True
        )
        
        if result and len(result) > 0:
            tei_xml = result[0]
            return self._parse_tei_xml(tei_xml)
        
        return None
    
    def _parse_tei_xml(self, tei_xml: str) -> Dict:
        
        soup = BeautifulSoup(tei_xml, 'xml')
        sections = {}
        
        abstract_elem = soup.find('abstract')
        if abstract_elem:
            sections['abstract'] = self._clean_text(abstract_elem.get_text())
        
        intro_elem = soup.find('div', {'type': 'introduction'})
        if not intro_elem:
            intro_elem = soup.find('div', {'type': 'background'})
        if intro_elem:
            sections['introduction'] = self._clean_text(intro_elem.get_text())
        
        concl_elem = soup.find('div', {'type': 'conclusion'})
        if not concl_elem:
            concl_elem = soup.find('div', {'type': 'discussion'})
        if concl_elem:
            sections['conclusion'] = self._clean_text(concl_elem.get_text())
        
        return sections
    
    def extract_with_pymupdf(self, pdf_path: Path) -> Dict:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        
        sections = self._extract_sections_simple(full_text)
        return sections
    
    def _extract_sections_simple(self, text: str) -> Dict:
        sections = {}
        text_normalized = re.sub(r'\s+', ' ', text)
        
        abstract_patterns = [
            r'(?:abstract|summary)\s*:?\s*(.*?)(?:\s+(?:introduction|background|keywords|1\.|introduction|background))',
            r'abstract\s+(.*?)(?:\s+introduction)',
            r'summary\s+(.*?)(?:\s+introduction)',
        ]
        
        for pattern in abstract_patterns:
            abstract_match = re.search(pattern, text_normalized, re.IGNORECASE | re.DOTALL)
            if abstract_match:
                abstract_text = abstract_match.group(1).strip()
                words = abstract_text.split()
                if len(words) > 500:
                    abstract_text = ' '.join(words[:500])
                sections['abstract'] = self._clean_text(abstract_text)
                break
        
        intro_patterns = [
            r'(?:introduction|background)\s*:?\s*(.*?)(?:\s+(?:methods?|materials|methodology|2\.|results|materials\s+and\s+methods))',
            r'1\.\s*(?:introduction|background)\s*(.*?)(?:\s+2\.)',
        ]
        
        for pattern in intro_patterns:
            intro_match = re.search(pattern, text_normalized, re.IGNORECASE | re.DOTALL)
            if intro_match:
                intro_text = intro_match.group(1).strip()
                words = intro_text.split()
                if len(words) > 2000:
                    intro_text = ' '.join(words[:2000])
                sections['introduction'] = self._clean_text(intro_text)
                break
        
        concl_patterns = [
            r'(?:conclusion|discussion|conclusions?)\s*:?\s*(.*?)(?:\s+(?:references?|acknowledgments?|acknowledgements?|conflict|author|funding|$))',
            r'(?:conclusion|discussion)\s+(.*?)(?:\s+references)',
        ]
        
        for pattern in concl_patterns:
            concl_match = re.search(pattern, text_normalized, re.IGNORECASE | re.DOTALL)
            if concl_match:
                concl_text = concl_match.group(1).strip()
                words = concl_text.split()
                if len(words) > 1500:
                    concl_text = ' '.join(words[:1500])
                sections['conclusion'] = self._clean_text(concl_text)
                break
        
        return sections
    
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\/\°\±\×\÷\α\β\γ\δ\ε\μ\σ\τ]', '', text)
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r',{2,}', ',', text)
        
        return text.strip()
    
    def process_pdf(self, pdf_path: Path, doc_metadata: Dict) -> Optional[Dict]:
        sections = self.extract_with_grobid(pdf_path)
        
        if not sections or not any(sections.values()):
            sections = self.extract_with_pymupdf(pdf_path)
        
        if not sections or not any(sections.values()):
            return None
        
        document = {
            "doc_id": doc_metadata.get('doc_id', pdf_path.stem),
            "title": doc_metadata.get('title', ''),
            "year": doc_metadata.get('year'),
            "source": doc_metadata.get('source', 'pdf'),
            "url": doc_metadata.get('url', ''),
            "pmid": doc_metadata.get('pmid', ''),
            "doi": doc_metadata.get('doi', ''),
            "sections": sections
        }
        
        return document
    
    def process_all_pdfs(self, papers_dir: Optional[Path] = None):
        if papers_dir is None:
            papers_dir = self.papers_dir
        
        if not papers_dir.exists():
            return
        
        manifest_path = Path("data/papers_manifest.csv")
        metadata_map = {}
        if manifest_path.exists():
            import csv
            with open(manifest_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metadata_map[row['doc_id']] = row
        
        pdf_files = list(papers_dir.glob("*.pdf"))
        
        if len(pdf_files) == 0:
            return
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for pdf_path in pdf_files:
                doc_id = pdf_path.stem
                metadata = metadata_map.get(doc_id, {
                    'doc_id': doc_id,
                    'title': pdf_path.stem,
                    'source': 'pdf',
                    'year': None
                })
                
                document = self.process_pdf(pdf_path, metadata)
                
                if document and document.get('sections'):
                    has_content = any(
                        section_text and len(section_text.strip()) > 50 
                        for section_text in document['sections'].values()
                    )
                    
                    if has_content:
                        f.write(json.dumps(document, ensure_ascii=False) + '\n')


