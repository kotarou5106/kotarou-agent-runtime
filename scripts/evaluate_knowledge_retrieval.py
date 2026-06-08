from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_system.evaluation.dataset import load_eval_dataset
from knowledge_system.evaluation.report import write_json_report, write_markdown_report
from knowledge_system.evaluation.runner import run_retrieval_evaluation
from knowledge_system.indexing.store import KnowledgeStore
from knowledge_system.retrieval import KnowledgeRetriever

_DEFAULT_DATASET = Path("knowledge_system/evaluation/fixtures/sample_eval_dataset.json")
_MODES = ("bm25", "vector", "hybrid_rrf")


async def _run(args: argparse.Namespace) -> int:
    dataset = load_eval_dataset(args.dataset)
    modes = list(_MODES) if args.mode == "all" else [args.mode]
    store = KnowledgeStore(Path(args.workspace) / "knowledge.db")
    try:
        retriever = KnowledgeRetriever(store=store, embedder=None, top_k=args.top_k)
        results = [
            await run_retrieval_evaluation(
                dataset,
                retriever,
                mode=mode,
                top_k=args.top_k,
                include_trace=args.include_trace,
            )
            for mode in modes
        ]
    finally:
        store.close()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "knowledge_retrieval_eval.json"
    markdown_path = output_dir / "knowledge_retrieval_eval.md"
    write_json_report(results, json_path)
    write_markdown_report(results, markdown_path)
    _print_summary(results, json_path=json_path, markdown_path=markdown_path)
    return 0


def _print_summary(results, *, json_path: Path, markdown_path: Path) -> None:
    print("Knowledge Retrieval Evaluation")
    print("Retriever        Recall@1  Recall@3  Recall@5  Precision@5  HitRate@5  MRR@5  NDCG@5")
    for result in results:
        metrics = result.metrics
        print(
            f"{result.retriever_name:<16} "
            f"{metrics.get('recall@1', 0.0):>8.3f} "
            f"{metrics.get('recall@3', 0.0):>8.3f} "
            f"{metrics.get('recall@5', 0.0):>8.3f} "
            f"{metrics.get('precision@5', 0.0):>11.3f} "
            f"{metrics.get('hit_rate@5', 0.0):>9.3f} "
            f"{metrics.get('mrr@5', 0.0):>6.3f} "
            f"{metrics.get('ndcg@5', 0.0):>7.3f}"
        )
    print(f"\nJSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    print("Note: vector mode requires a configured embedder in production wiring; this script uses the local store only.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Knowledge RAG retrieval quality.")
    parser.add_argument("--dataset", default=str(_DEFAULT_DATASET), help="Path to retrieval eval dataset JSON.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve per case.")
    parser.add_argument("--output-dir", default="reports", help="Directory for JSON and Markdown reports.")
    parser.add_argument("--workspace", default=".", help="Workspace directory containing knowledge.db.")
    parser.add_argument("--mode", choices=("bm25", "vector", "hybrid_rrf", "all"), default="all")
    parser.add_argument("--no-trace", dest="include_trace", action="store_false", help="Disable retrieval traces.")
    parser.set_defaults(include_trace=True)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
