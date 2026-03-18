import streamlit as st
import os
from pathlib import Path
from src.rag.retriever import Retriever
from src.rag.reranker import Reranker
from src.rag.generator import LLMGenerator
from src.storage.vector_store import VectorStore
from src.embedding.embedder import BaseEmbedder, NeuMLEmbedder
from articles_mapping import get_article_info
import yaml
from dotenv import load_dotenv


st.set_page_config(page_title="RAG Agent - Alzheimer Targets", layout="wide")

@st.cache_data
def load_config():
    with open("config.yaml", 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@st.cache_resource
def init_components():
    embedder = NeuMLEmbedder()
    vector_store = VectorStore()
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    reranker = Reranker()
    generator = LLMGenerator()
    return retriever, reranker, generator

def is_no_information_answer(answer: str) -> bool:
    if not answer:
        return True
    no_info_phrases = [
        "doesn't contain information",
        "doesn't contain",
        "no information",
        "not enough information",
    ]
    return any(phrase in answer.lower() for phrase in no_info_phrases)

st.title("RAG-агент")

config = load_config()
rag_config = config.get('rag', {})
llm_config = config.get('llm', {})
provider = llm_config.get('provider', 'groq').lower()

required_key = {'groq': 'GROQ_API_KEY', 'openai': 'OPENAI_API_KEY'}.get(provider)
if required_key and required_key not in os.environ:
    st.warning(f"{required_key} не установлен.")

retriever, reranker, generator = init_components()

with st.sidebar:
    st.header("Параметры поиска")
    enable_reranking = st.checkbox("Включить reranking", value=True)

    filter_year_input = st.text_input("Фильтр по году", value="", placeholder="например: 2024")
    filter_year = int(filter_year_input) if filter_year_input.strip().isdigit() else 0

    st.divider()
    st.header("Статус")
    if Path("data/chunks.jsonl").exists():
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
        filters = {'year': str(filter_year)} if filter_year > 0 else None

        retrieved_chunks = retriever.retrieve(
            query,
            top_k=rag_config.get('top_k_retrieval', 30),
            filters=filters,
        )

        if not retrieved_chunks:
            answer = "RAG система не нашла подходящих документов для вашего запроса."
        else:
            top_k_rerank = rag_config.get('top_k_rerank', 8)
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

        # одна статья — один источник
        seen_docs = {}
        for chunk in retrieved_chunks:
            doc_id = chunk.get('metadata', {}).get('doc_id') or chunk.get('chunk_id', '').split('_')[0]
            if doc_id not in seen_docs:
                seen_docs[doc_id] = chunk
        unique_chunks = list(seen_docs.values())

        for i, chunk in enumerate(unique_chunks, 1):
            meta = chunk.get('metadata', {})
            doc_id = meta.get('title', '') or chunk.get('doc_id', '') or chunk.get('chunk_id', '').split('_')[0]

            article_info = get_article_info(doc_id)
            if article_info:
                article_title = article_info['title']
                article_url = article_info['url']
            else:
                article_title = meta.get('title', 'Unknown Article')
                article_url = meta.get('url', '')

            year = meta.get('year', '')
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
