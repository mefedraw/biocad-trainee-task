import streamlit as st
import os
from pathlib import Path
from typing import List, Dict, Optional


from dotenv import load_dotenv
root_dir = Path(__file__).parent
env_path = root_dir / '.env'
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)


from src.rag.retriever import Retriever
from src.rag.reranker import Reranker
from src.rag.generator import LLMGenerator
from src.rag.vector_store import VectorStore
from src.embedding.embedder import Embedder
from articles_mapping import get_article_info
import yaml


st.set_page_config(page_title="RAG Agent - Alzheimer Targets", layout="wide")

@st.cache_data
def load_config():
    with open("config.yaml", 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@st.cache_resource
def init_components():
    embedder = Embedder()
    vector_store = VectorStore()
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    reranker = Reranker()
    generator = LLMGenerator()
    return retriever, reranker, generator

def is_no_information_answer(answer: str) -> bool:
    if not answer:
        return True
    
    answer_lower = answer.lower()
    no_info_phrases = [
        "does not contain information",
        "doesn't contain information",
        "does not contain",
        "doesn't contain",
        "no information",
        "not enough information",
        "cannot answer",
        "unable to answer",
        "cannot provide",
        "unable to provide",
        "context doesn't contain",
        "context does not contain",
        "provided context does not",
        "provided context doesn't",
        "the provided context does not",
        "the provided context doesn't",
        "not contain information to answer",
        "doesn't have information",
        "does not have information"
    ]
    
    return any(phrase in answer_lower for phrase in no_info_phrases)

st.title("RAG-агент")

config = load_config()
llm_config = config.get('llm', {})
provider = llm_config.get('provider', 'openai').lower()
rag_config = config.get('rag', {})
max_distance = rag_config.get('max_distance', 1.2)

api_keys = {
    'openai': 'OPENAI_API_KEY',
    'groq': 'GROQ_API_KEY',
    'ollama': None
}

required_key = api_keys.get(provider)
if required_key and required_key not in os.environ:
    st.warning(f"{required_key} не установлен.")

retriever, reranker, generator = init_components()

with st.sidebar:
    st.header("Параметры поиска")
    top_k_retrieval = st.slider("Количество чанков для retrieval", 10, 50, 30)
    top_k_rerank = st.slider("Количество чанков после reranking", 5, 20, 8)
    enable_reranking = st.checkbox("Включить reranking", value=True)
    
    filter_year_input = st.text_input("Фильтр по году (пусто = без фильтра)", value="", placeholder="например: 2024")
    filter_year = int(filter_year_input) if filter_year_input.strip() and filter_year_input.strip().isdigit() else 0
    
    st.divider()
    st.header("Статус")
    chunks_path = Path("data/chunks.jsonl")
    if chunks_path.exists():
        st.success("Данные загружены")
    else:
        st.warning("Данные не найдены. Запустите preprocessing pipeline.")

query = st.text_input(
    "Введите ваш вопрос:",
    placeholder="Например: What are potential targets for Alzheimer's disease treatment?",
    key="query_input"
)

col1, col2 = st.columns([1, 4])
with col1:
    search_button = st.button("Поиск", type="primary", use_container_width=True)

if search_button and query:
    answer = None
    retrieved_chunks = []
    
    with st.spinner("Поиск релевантных документов..."):
        filters = None
        if filter_year > 0:
            filters = {'year': str(filter_year)}
        
        retrieved_chunks = retriever.retrieve(query, top_k=top_k_retrieval, filters=filters)
        
        if not retrieved_chunks:
            answer = "RAG система не нашла подходящих документов для вашего запроса."
            retrieved_chunks = []
        else:
            if enable_reranking and reranker:
                retrieved_chunks = reranker.rerank(query, retrieved_chunks, top_k=top_k_rerank)
            else:
                retrieved_chunks = retrieved_chunks[:top_k_rerank]
            
            if retrieved_chunks:
                answer = generator.generate(query, retrieved_chunks)
            else:
                answer = "RAG система не нашла подходящих документов для вашего запроса."
    
    st.divider()
    st.header("Ответ")
    if answer.startswith("Ошибка"):
        st.error(answer)
    else:
        st.markdown(answer)
    
    if not is_no_information_answer(answer) and retrieved_chunks:
        st.divider()
        st.header("Источники")
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            meta = chunk.get('metadata', {})
            doc_id = meta.get('title', '') or chunk.get('doc_id', '') or chunk.get('chunk_id', '').split('_')[0]
            
            article_info = get_article_info(doc_id)
            if article_info:
                article_title = article_info['title']
                article_url = article_info['url']
            else:
                article_title = meta.get('title', 'Unknown Article')
                article_url = meta.get('url', '')
            
            year = meta.get('year', '') or meta.get('year', None)
            section = meta.get('section', '')
            pmid = meta.get('pmid', '')
            
            header_text = f"[{i}] {article_title}"
            if year and year != 'None' and str(year).strip():
                header_text += f" ({year})"
            
            with st.expander(header_text):
                col1, col2 = st.columns([3, 1])
                with col1:
                    if section:
                        st.write(f"**Секция:** {section}")
                    if article_url:
                        st.markdown(f"**Ссылка:** [{article_url}]({article_url})")
                    if pmid:
                        st.write(f"**PMID:** {pmid}")
                with col2:
                    if chunk.get('distance'):
                        st.metric("Релевантность", f"{1 - chunk['distance']:.3f}")
                    if meta.get('rerank_score'):
                        st.metric("Rerank Score", f"{meta['rerank_score']:.2f}")
                
                st.write("**Текст:**")
                st.text(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])

with st.expander("Примеры вопросов"):
    example_questions = [
        "What are potential targets for Alzheimer's disease treatment?",
        "Are the targets druggable with small molecules, biologics, or other modalities?",
        "What additional studies are needed to advance these targets?",
        "What is the role of amyloid beta in Alzheimer's disease?",
        "Which tau protein modifications are associated with Alzheimer's?"
    ]
    
    for q in example_questions:
        if st.button(q, key=f"example_{q}", use_container_width=True):
            st.session_state.query_input = q
            st.rerun()

