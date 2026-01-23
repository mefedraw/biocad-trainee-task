from typing import List, Dict, Set, Tuple
import numpy as np
import re


class EvaluationMetrics:    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
        
        top_k_retrieved = set(retrieved_ids[:k])
        relevant_retrieved = top_k_retrieved.intersection(relevant_ids)
        
        return len(relevant_retrieved) / len(relevant_ids)
    
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        if k == 0:
            return 0.0
        
        top_k_retrieved = set(retrieved_ids[:k])
        relevant_retrieved = top_k_retrieved.intersection(relevant_ids)
        
        return len(relevant_retrieved) / k
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def evaluate_retrieval(test_questions: List[Dict], retriever, top_k: int = 10) -> Dict:
        recalls = []
        precisions = []
        mrrs = []
        
        for question_data in test_questions:
            query = question_data['question']
            relevant_ids = set(question_data.get('relevant_doc_ids', []))
            
            if not relevant_ids:
                continue
            
            results = retriever.retrieve(query, top_k=top_k)
            retrieved_ids = [r['chunk_id'] for r in results]
            
            recall = EvaluationMetrics.recall_at_k(retrieved_ids, relevant_ids, top_k)
            precision = EvaluationMetrics.precision_at_k(retrieved_ids, relevant_ids, top_k)
            mrr = EvaluationMetrics.mean_reciprocal_rank(retrieved_ids, relevant_ids)
            
            recalls.append(recall)
            precisions.append(precision)
            mrrs.append(mrr)
        
        return {
            'recall@k': np.mean(recalls) if recalls else 0.0,
            'precision@k': np.mean(precisions) if precisions else 0.0,
            'mrr': np.mean(mrrs) if mrrs else 0.0,
            'num_questions': len(test_questions)
        }
    
    @staticmethod
    def evaluate_generation(answer: str, retrieved_chunks: List[Dict]) -> Dict:
        sentences = EvaluationMetrics._split_sentences(answer)
        if not sentences:
            return {"faithfulness": 0.0, "citation_accuracy": 0.0}

        citation_map = EvaluationMetrics._build_citation_map(retrieved_chunks)
        if not citation_map:
            return {"faithfulness": 0.0, "citation_accuracy": 0.0}

        sentence_results = []
        citation_hits = []

        for sentence in sentences:
            cited_nums = EvaluationMetrics._extract_citations(sentence)
            if not cited_nums:
                sentence_results.append(False)
                continue

            supported = False
            for num in cited_nums:
                chunk_text = citation_map.get(num, "")
                if chunk_text and EvaluationMetrics._has_support(sentence, chunk_text):
                    supported = True
                    citation_hits.append(True)
                else:
                    citation_hits.append(False)
            sentence_results.append(supported)

        faithfulness = float(np.mean(sentence_results)) if sentence_results else 0.0
        citation_accuracy = float(np.mean(citation_hits)) if citation_hits else 0.0

        return {"faithfulness": faithfulness, "citation_accuracy": citation_accuracy}

    @staticmethod
    def _build_citation_map(retrieved_chunks: List[Dict]) -> Dict[int, str]:
        citation_map = {}
        for i, chunk in enumerate(retrieved_chunks, start=1):
            citation_map[i] = chunk.get("text", "")
        return citation_map

    @staticmethod
    def _extract_citations(text: str) -> List[int]:
        matches = re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text)
        nums = []
        for match in matches:
            for n in match.split(","):
                n = n.strip()
                if n.isdigit():
                    nums.append(int(n))
        return nums

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        tokens = re.findall(r"[a-zA-Z0-9\-]+", text.lower())
        return set(tokens)

    @staticmethod
    def _has_support(sentence: str, chunk_text: str, min_overlap: float = 0.08) -> bool:
        sent_tokens = EvaluationMetrics._tokenize(sentence)
        if not sent_tokens:
            return False
        chunk_tokens = EvaluationMetrics._tokenize(chunk_text)
        if not chunk_tokens:
            return False
        overlap = sent_tokens.intersection(chunk_tokens)
        return (len(overlap) / len(sent_tokens)) >= min_overlap