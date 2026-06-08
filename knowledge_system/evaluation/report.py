from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from knowledge_system.evaluation.dataset import RetrievalEvalResult


def write_json_report(results: list[RetrievalEvalResult], path: str | Path) -> None:
    """Write retrieval evaluation results to JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": [result.to_dict() for result in results],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def render_markdown_report(results: list[RetrievalEvalResult]) -> str:
    """Render a compact Markdown report for retrieval evaluation."""
    lines = [
        "# Knowledge Retrieval Evaluation Report",
        "",
        "## Summary",
        "",
        "| Retriever | Recall@1 | Recall@3 | Recall@5 | Precision@5 | HitRate@5 | MRR@5 | NDCG@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        metrics = result.metrics
        lines.append(
            "| {name} | {r1:.3f} | {r3:.3f} | {r5:.3f} | {p5:.3f} | {h5:.3f} | {mrr:.3f} | {ndcg:.3f} |".format(
                name=result.retriever_name,
                r1=metrics.get("recall@1", 0.0),
                r3=metrics.get("recall@3", 0.0),
                r5=metrics.get("recall@5", 0.0),
                p5=metrics.get("precision@5", 0.0),
                h5=metrics.get("hit_rate@5", 0.0),
                mrr=metrics.get("mrr@5", 0.0),
                ndcg=metrics.get("ndcg@5", 0.0),
            )
        )

    lines.extend(["", "## Per-case Results", ""])
    for result in results:
        lines.extend([f"### {result.retriever_name}", ""])
        for case in result.case_results:
            relevant = case.relevant_chunk_ids or case.relevant_document_ids
            retrieved = case.retrieved_chunk_ids or case.retrieved_document_ids
            hit = "hit" if case.hit_at_k.get("5", False) else "miss"
            rank = case.first_relevant_rank if case.first_relevant_rank is not None else "-"
            lines.extend(
                [
                    f"- `{case.case_id}` {hit}",
                    f"  - query: {case.query}",
                    f"  - relevant ids: {', '.join(relevant) if relevant else '-'}",
                    f"  - retrieved ids: {', '.join(retrieved) if retrieved else '-'}",
                    f"  - first relevant rank: {rank}",
                    f"  - trace_id: {case.trace_id or '-'}",
                ]
            )
            if case.error:
                lines.append(f"  - error: {case.error}")
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- This is a lightweight internal retrieval evaluation.",
            "- It evaluates retrieval quality only, not final answer generation.",
            "- Answer faithfulness and hallucination evaluation can be added later.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(results: list[RetrievalEvalResult], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown_report(results), encoding="utf-8")


def result_to_dict(result: RetrievalEvalResult) -> dict[str, Any]:
    return asdict(result)
