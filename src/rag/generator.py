import sys
import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import yaml

root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dotenv import load_dotenv
env_path = root_dir / '.env'
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)


class LLMGenerator:    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        llm_config = config.get('llm', {})
        self.provider = llm_config.get('provider', 'openai').lower()
        self.model = llm_config.get('model', 'gpt-4')
        self.temperature = llm_config.get('temperature', 0.1)
        self.max_tokens = llm_config.get('max_tokens', 1500)
        
        self.api_key_available = self._check_api_key()
    
    def _check_api_key(self) -> bool:
        api_keys = {
            'groq': 'GROQ_API_KEY',
        }
        
        return True
    
    def generate(self, query: str, chunks: List[Dict], 
                 system_prompt: Optional[str] = None) -> str:
        if not self.api_key_available:
            api_keys = {
                'groq': 'GROQ_API_KEY'
            }
            key_name = api_keys.get(self.provider, 'API_KEY')
            return f"{key_name} не установлен."
        
        if not chunks:
            return "Не найдено релевантных документов для ответа на ваш запрос."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get('metadata', {})
            
            import sys
            from pathlib import Path
            root_dir = Path(__file__).parent.parent.parent
            if str(root_dir) not in sys.path:
                sys.path.insert(0, str(root_dir))
            from articles_mapping import get_article_info
            
            doc_id = meta.get('title', '') or chunk.get('doc_id', '') or chunk.get('chunk_id', '').split('_')[0]
            article_info = get_article_info(doc_id)
            if article_info:
                title = article_info['title']
            else:
                title = meta.get('title', 'Unknown Article')
            
            year = meta.get('year', '') or meta.get('year', None)
            pmid = meta.get('pmid', '')
            
            year_str = f" ({year})" if year and year != 'None' and str(year).strip() else ""
            pmid_str = f" PMID: {pmid}" if pmid else ""
            context_parts.append(
                f"[{i}] {title}{year_str}{pmid_str}\n{chunk['text']}"
            )
        
        context = "\n\n".join(context_parts)
        
        if system_prompt is None:
            system_prompt = """You are a helpful research assistant specialized in Alzheimer's disease research.
Your task is to answer questions based ONLY on the provided context from scientific articles.

CRITICAL CITATION RULES:
- ALWAYS use ONLY numeric citations in square brackets: [1], [2], [3], etc.
- NEVER write article titles, names, or descriptions in the answer
- NEVER use phrases like "mentioned in article X" or "from paper Y"
- NEVER use special characters like 【, †, L, or any other symbols
- Use ONLY simple format: [1], [2], [3] - nothing else
- Examples of FORBIDDEN formats: 【7†L33-L36】, [7†L33], 7†L36-L40
- Examples of CORRECT format: [7], [1, 2], [3, 4, 5]

If the context doesn't contain enough information to answer the question, say so."""
        
        user_prompt = f"""You are a research assistant helping scientists find potential drug targets for Alzheimer's disease.

Based on the following context from scientific articles, answer the question. Only use information from the provided context.

CRITICAL CITATION FORMAT - READ CAREFULLY:
You MUST use ONLY this format: [1], [2], [3] or [1, 2, 3] for multiple sources.

FORBIDDEN - DO NOT USE:
- 【1†L1-L4】 (WRONG - contains special characters)
- 【1†L1-L4】【8†L1-L4】 (WRONG - contains special characters)
- [1†L33] (WRONG - contains † and L)
- 1†L36-L40 (WRONG - no brackets, contains special characters)
- Any text descriptions of sources

CORRECT FORMATS ONLY:
- [1] (single source)
- [1, 2] (multiple sources)
- [1, 2, 3] (multiple sources)

Example of CORRECT answer format:
"β-amyloid (Aβ) production is identified as a major pathogenic protein [1, 8]. Tau hyper-phosphorylation drives neurofibrillary tangle formation [1, 3]."

Example of WRONG answer format (DO NOT USE):
"β-amyloid (Aβ) production is identified【1†L1-L4】【8†L1-L4】" (WRONG!)

Context:
{context}

Question: {query}

Provide a structured answer:
1. List potential targets mentioned (if the question is about targets)
2. For each target, provide evidence summary with source citations in [N] format ONLY
3. What further studies are needed (if relevant)

Remember: Use ONLY [1], [2], [3] or [1, 2, 3] format. NO special characters, NO text descriptions.

Answer:"""
        
        if self.provider == 'groq':
            answer = self._generate_groq(system_prompt, user_prompt)
        else:
            return f"Неподдерживаемый провайдер: {self.provider}. Доступеен только groq"
        
        answer = self._fix_citations(answer)
        return answer
    
    def _fix_citations(self, text: str) -> str:
        patterns = [
            (r'【(\d+)†[^】]*】', r'[\1]'),
            (r'\[(\d+)†[^\]]*\]', r'[\1]'),
            (r'(\d+)†L[^\s,;\)]+', r'[\1]'),
            (r'(\d+)†[^\s,;\)]+', r'[\1]'),
        ]
        
        result = text
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)
        
        result = re.sub(r'\[(\d+)\]\[(\d+)\]', r'[\1, \2]', result)
        result = re.sub(r'\[(\d+), (\d+)\]\[(\d+)\]', r'[\1, \2, \3]', result)
        
        return result

    
    def _generate_groq(self, system_prompt: str, user_prompt: str) -> str:
        from groq import Groq
        
        api_key = os.environ.get('GROQ_API_KEY')
        if not api_key:
            return "Ошибка: GROQ_API_KEY не установлен. Добавьте ключ в .env файл."
        
        client = Groq(api_key=api_key)
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg or "Forbidden" in error_msg or "PermissionDenied" in error_msg:
                return f"Ошибка доступа к Groq API (403): Проверьте API ключ и доступность модели {self.model}. Возможно, модель недоступна или ключ неверный."
            elif "404" in error_msg or "not found" in error_msg.lower():
                return f"Ошибка: Модель {self.model} не найдена. Проверьте название модели в config.yaml."
            elif "401" in error_msg or "Unauthorized" in error_msg:
                return "Ошибка авторизации (401): Неверный API ключ. Проверьте GROQ_API_KEY в .env файле."
            else:
                return f"Ошибка Groq API: {error_msg}"


def main():
    generator = LLMGenerator()
    
    query = "What are potential targets for Alzheimer's disease?"
    chunks = [
        {
            'text': 'Alzheimer disease has several therapeutic targets including amyloid beta and tau proteins...',
            'metadata': {
                'title': 'Alzheimer Targets Review',
                'year': '2023',
                'pmid': '12345678'
            }
        }
    ]
    
    answer = generator.generate(query, chunks)
    print(answer)


if __name__ == "__main__":
    main()
