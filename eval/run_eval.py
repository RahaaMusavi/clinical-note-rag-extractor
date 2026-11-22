import json
import time
from dotenv import load_dotenv

load_dotenv(override=True)

from src.retriever import load_guidelines, GuidelineRetriever
from src.extractor import ClinicalNoteExtractor


def evaluate_pipeline(
    eval_data: list,
    retriever: GuidelineRetriever,
    extractor: ClinicalNoteExtractor,
    mode: str = "hybrid",
):
    retrieval_hits = 0
    answerable_total = 0
    citation_exact = 0
    unanswerable_total = 0
    false_positives = 0
    latencies = []

    for item in eval_data:
        t0 = time.perf_counter()
        chunks = (
            retriever.retrieve_dense_only(item["input_note"], top_k=3)
            if mode == "dense"
            else retriever.retrieve_hybrid_reranked(item["input_note"], top_k=3)
        )
        retrieved_ids = {c.rule_id for c in chunks}

        if not item["should_abstain"]:
            answerable_total += 1
            if set(item["expected_rule_ids"]).issubset(retrieved_ids):
                retrieval_hits += 1

        prediction = extractor.extract(item["input_note"], chunks)
        latencies.append((time.perf_counter() - t0) * 1000)

        if item["should_abstain"]:
            unanswerable_total += 1
            if not prediction.abstained or len(prediction.cited_rule_ids) > 0:
                false_positives += 1
        else:
            if set(prediction.cited_rule_ids) == set(item["expected_rule_ids"]):
                citation_exact += 1

    return {
        "recall_at_3": retrieval_hits / max(answerable_total, 1),
        "citation_acc": citation_exact / max(answerable_total, 1),
        "hallucination_rate": false_positives / max(unanswerable_total, 1),
        "avg_latency_ms": sum(latencies) / max(len(latencies), 1),
    }


if __name__ == "__main__":
    chunks = load_guidelines("data/guidelines.md")
    retriever = GuidelineRetriever(chunks)
    extractor = ClinicalNoteExtractor(model="gpt-4o-mini")

    with open("data/eval_set.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    dense_metrics = evaluate_pipeline(eval_data, retriever, extractor, mode="dense")
    hybrid_metrics = evaluate_pipeline(eval_data, retriever, extractor, mode="hybrid")

    print("\n| Pipeline Variant | Retrieval Recall@3 | Rule Citation Exact Match | Unanswerable Hallucination Rate | Avg Latency |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    print(
        f"| Dense Vector RAG (Baseline) | {dense_metrics['recall_at_3']:.1%} | "
        f"{dense_metrics['citation_acc']:.1%} | {dense_metrics['hallucination_rate']:.1%} | "
        f"{dense_metrics['avg_latency_ms']:.0f} ms |"
    )
    print(
        f"| Hybrid (BM25 + Dense) + Reranker + Schema Guard | {hybrid_metrics['recall_at_3']:.1%} | "
        f"{hybrid_metrics['citation_acc']:.1%} | {hybrid_metrics['hallucination_rate']:.1%} | "
        f"{hybrid_metrics['avg_latency_ms']:.0f} ms |"
    )
