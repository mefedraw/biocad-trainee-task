import sys
import json
import argparse
from pathlib import Path

root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.rag.retriever import Retriever
from src.rag.reranker import Reranker
from src.rag.generator import LLMGenerator
from src.rag.vector_store import VectorStore
from src.embedding.embedder import Embedder
from src.evaluation.metrics import EvaluationMetrics


def load_questions(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="src/evaluation/test_questions.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rerank-top-k", type=int, default=8)
    parser.add_argument("--with-generation", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        return

    test_questions = load_questions(questions_path)
    if args.limit and args.limit > 0:
        test_questions = test_questions[: args.limit]

    embedder = Embedder()
    vector_store = VectorStore()
    retriever = Retriever(vector_store=vector_store, embedder=embedder)

    retrieval_metrics = EvaluationMetrics.evaluate_retrieval(
        test_questions, retriever, top_k=args.top_k
    )
    print(retrieval_metrics)

    if not args.with_generation:
        return

    reranker = Reranker()
    generator = LLMGenerator()

    faithfulness_scores = []
    citation_scores = []

    for item in test_questions:
        query = item["question"]
        retrieved = retriever.retrieve(query, top_k=args.top_k)
        retrieved = reranker.rerank(query, retrieved, top_k=args.rerank_top_k)
        answer = generator.generate(query, retrieved)
        gen_metrics = EvaluationMetrics.evaluate_generation(answer, retrieved)
        faithfulness_scores.append(gen_metrics["faithfulness"])
        citation_scores.append(gen_metrics["citation_accuracy"])

    gen_summary = {
        "faithfulness": float(sum(faithfulness_scores) / len(faithfulness_scores))
        if faithfulness_scores
        else 0.0,
        "citation_accuracy": float(sum(citation_scores) / len(citation_scores))
        if citation_scores
        else 0.0,
        "num_questions": len(test_questions),
    }
    print(gen_summary)


if __name__ == "__main__":
    main()
