#!/usr/bin/env python3
"""Build ToolRet workloads whose menu membership comes from retrieval.

Unlike ``build_cluster_workload.py --partition toolret``, this script never
places gold relevance labels into the request menu. Gold IDs are read only
after retrieval to score recall/precision and are retained as provenance for
offline evaluation. Ordering is applied to the retrieved top-k set, keeping
retrieval error separate from cache/ordering behavior as required by the brief.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tatm.analysis import deduplicated_existing_ids, load_processed
from tatm.io import write_json, write_jsonl
from tatm.models import TaskRecord
from tatm.prompting import order_tool_ids, workload_record
from tatm.retrieval import (
    BM25ToolRetriever,
    aggregate_retrieval_metrics,
    retrieval_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a label-free BM25-selected ToolRet workload."
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Defaults to <output stem>-retrieval-metrics.json.",
    )
    parser.add_argument("--menu-size", type=int, default=64)
    parser.add_argument(
        "--report-k",
        type=int,
        nargs="+",
        default=(4, 16, 64, 128),
        help=(
            "Also report a retrieval curve at these cutoffs. The deepest cutoff "
            "is retrieved once; only --menu-size is sent in the workload."
        ),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--ordering",
        choices=(
            "original",
            "alphabetical",
            "random",
            "frequency",
            "schema_cost_weighted",
            "fp_tree_global",
        ),
        default="original",
        help="Applied after BM25 selects the menu; original preserves BM25 rank.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument(
        "--support-mode",
        choices=("disjoint", "preceding", "all"),
        default="disjoint",
        help=(
            "Gold-label split used only to fit frequency-based ordering. "
            "It never affects which tools BM25 retrieves."
        ),
    )
    args = parser.parse_args()

    if args.menu_size < 1:
        parser.error("--menu-size must be >= 1")
    if any(value < 1 for value in args.report_k):
        parser.error("every --report-k value must be >= 1")
    if args.offset < 0:
        parser.error("--offset must be >= 0")
    if args.limit < 1:
        parser.error("--limit must be >= 1")

    tools, tasks = load_processed(args.processed_dir)
    corpus = {
        tool_id: tool
        for tool_id, tool in tools.items()
        if tool.source.startswith("toolret:")
    }
    benchmark_tasks = [
        task for task in tasks if task.evidence_type == "gold_relevance"
    ]
    evaluation_tasks = benchmark_tasks[args.offset : args.offset + args.limit]
    if not corpus:
        raise SystemExit("No ToolRet tools found in the processed corpus.")
    if not evaluation_tasks:
        raise SystemExit("The requested ToolRet evaluation slice is empty.")
    if args.menu_size > len(corpus):
        parser.error(
            f"--menu-size {args.menu_size} exceeds the {len(corpus)}-tool corpus"
        )
    if max(args.report_k, default=args.menu_size) > len(corpus):
        parser.error(
            "a --report-k cutoff exceeds the "
            f"{len(corpus)}-tool corpus"
        )

    evaluation_ids = {task.task_id for task in evaluation_tasks}
    if args.support_mode == "disjoint":
        support_tasks = [
            task for task in benchmark_tasks if task.task_id not in evaluation_ids
        ]
        support_provenance = "disjoint_toolret_gold_labels"
    elif args.support_mode == "preceding":
        support_tasks = benchmark_tasks[: args.offset]
        support_provenance = "chronologically_preceding_toolret_gold_labels"
    else:
        support_tasks = benchmark_tasks
        support_provenance = "all_toolret_gold_labels_including_evaluation"

    support: Counter[str] = Counter()
    for task in support_tasks:
        support.update(deduplicated_existing_ids(task, corpus))

    retriever = BM25ToolRetriever(corpus, k1=args.bm25_k1, b=args.bm25_b)
    records = []
    metric_rows = []
    report_cutoffs = sorted(set((*args.report_k, args.menu_size)))
    curve_metric_rows: dict[int, list[dict[str, int | float]]] = {
        cutoff: [] for cutoff in report_cutoffs
    }
    curve_fallback_queries: Counter[int] = Counter()
    retrieved_frequency: Counter[str] = Counter()
    fallback_queries = 0
    for task in evaluation_tasks:
        # This is the selection step. It receives only query text and the tool
        # corpus; task.tool_ids (gold labels) are not an input.
        result = retriever.retrieve(task.query, k=max(report_cutoffs))
        menu_ids = result.tool_ids[: args.menu_size]
        menu_scores = result.scores[: args.menu_size]
        menu_fallback_count = sum(score == 0.0 for score in menu_scores)
        gold_ids = deduplicated_existing_ids(task, corpus)
        row_metrics = retrieval_metrics(menu_ids, gold_ids)
        metric_rows.append(row_metrics)
        retrieved_frequency.update(menu_ids)
        fallback_queries += int(menu_fallback_count > 0)
        for cutoff in report_cutoffs:
            cutoff_ids = result.tool_ids[:cutoff]
            cutoff_scores = result.scores[:cutoff]
            curve_metric_rows[cutoff].append(
                retrieval_metrics(cutoff_ids, gold_ids)
            )
            curve_fallback_queries[cutoff] += int(
                any(score == 0.0 for score in cutoff_scores)
            )

        ordered_ids = order_tool_ids(
            menu_ids,
            corpus,
            support,
            args.ordering,
            random_seed=args.random_seed,
        )
        retrieved_task = TaskRecord(
            task_id=task.task_id,
            source=task.source,
            query=task.query,
            tool_ids=menu_ids,
            evidence_type="retrieved_menu",
            domain=task.domain,
            metadata=task.metadata,
        )
        record = workload_record(
            retrieved_task, ordered_ids, corpus, args.ordering
        )
        record.update(
            {
                "retriever": "bm25_canonical_tools_v1",
                "retrieval_rank_tool_ids": list(menu_ids),
                "retrieval_scores": list(menu_scores),
                "retrieval_scored_candidates": result.scored_candidates,
                "retrieval_fallback_count": menu_fallback_count,
                # Gold is evaluation metadata only. Replay drivers send only
                # messages/tools and therefore cannot expose this to the model.
                "gold_tool_ids": list(gold_ids),
                "retrieval_metrics": row_metrics,
                "ordering_support_provenance": support_provenance,
            }
        )
        records.append(record)

    count = write_jsonl(args.output, records)
    metrics_output = args.metrics_output or args.output.with_name(
        f"{args.output.stem}-retrieval-metrics.json"
    )
    top_retrieved = [
        {
            "tool_id": tool_id,
            "name": corpus[tool_id].name,
            "retrieved_frequency": frequency,
        }
        for tool_id, frequency in retrieved_frequency.most_common(100)
    ]
    metrics = {
        "format_version": 1,
        "workload": args.output.as_posix(),
        "selection_mode": "retrieved_menu",
        "gold_labels_used_for_selection": False,
        "retriever": {
            "name": "bm25_canonical_tools_v1",
            "k1": args.bm25_k1,
            "b": args.bm25_b,
            "document_fields": [
                "function_name_x3",
                "description",
                "parameter_names_and_descriptions",
            ],
            "corpus_tools": retriever.corpus_size,
            "vocabulary_terms": retriever.vocabulary_size,
            "average_document_terms": round(
                retriever.average_document_length, 4
            ),
        },
        "evaluation": {
            "offset": args.offset,
            "queries": count,
            "menu_size": args.menu_size,
            "ordering": args.ordering,
            "random_seed": args.random_seed,
            "queries_requiring_deterministic_zero_score_fallback": fallback_queries,
            **aggregate_retrieval_metrics(metric_rows),
        },
        "retrieval_curve": [
            {
                "k": cutoff,
                "queries_requiring_deterministic_zero_score_fallback": (
                    curve_fallback_queries[cutoff]
                ),
                **aggregate_retrieval_metrics(curve_metric_rows[cutoff]),
            }
            for cutoff in report_cutoffs
        ],
        "ordering_fit": {
            "support_mode": args.support_mode,
            "support_provenance": support_provenance,
            "support_tasks": len(support_tasks),
            "evaluation_overlap_tasks": len(
                evaluation_ids & {task.task_id for task in support_tasks}
            ),
        },
        "retrieved_frequency_top_100": top_retrieved,
    }
    write_json(metrics_output, metrics)
    print(f"Wrote {count} retrieved-menu requests to {args.output}")
    print(f"Wrote retrieval metrics to {metrics_output}")
    print(
        "Recall@{size}={recall:.2%}  hit@{size}={hit:.2%}  MRR={mrr:.4f}".format(
            size=args.menu_size,
            recall=metrics["evaluation"]["mean_recall"],
            hit=metrics["evaluation"]["hit_rate"],
            mrr=metrics["evaluation"]["mean_reciprocal_rank"],
        )
    )


if __name__ == "__main__":
    main()
