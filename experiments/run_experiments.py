"""Controlled comparative experiment runner for ODIN (Report Sections 5, 7, 8).

Runs the exact same set of programming tasks under identical model, timeout,
and iteration limits across 4 experimental configurations (with run_memory OFF):
  - Config A (`direct`):       One-shot direct code generation (max_iterations=1)
  - Config B (`no_rag`):       Iterative self-debugging without RAG
  - Config C (`adaptive_rag`): Iterative self-debugging with selective Adaptive RAG
  - Config D (`always_rag`):   Iterative self-debugging with always-on RAG (RQ4)

Usage:
    python -m experiments.run_experiments --dataset custom --max-iterations 3
    python -m experiments.run_experiments --dataset mbpp --limit 20
    python -m experiments.run_experiments --dataset humaneval --limit 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from odin.backend import OdinBackend
from odin.config import config
from odin.state import ExperimentMode, RunHistory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_EXP_DIR = Path(__file__).resolve().parent
_DEFAULT_BENCHMARK = _EXP_DIR / "benchmark_suite.json"
_RESULTS_DIR = _EXP_DIR / "results"

CONFIG_MAP: dict[str, ExperimentMode] = {
    "config_a_direct": "direct",
    "config_b_no_rag": "no_rag",
    "config_c_adaptive_rag": "adaptive_rag",
    "config_d_always_rag": "always_rag",
}


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_custom_benchmark(path: Path = _DEFAULT_BENCHMARK) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mbpp_benchmark(limit: int = 20) -> list[dict[str, Any]]:
    """Load MBPP tasks via HuggingFace datasets."""
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=f"test[:{limit}]")
    tasks: list[dict[str, Any]] = []
    for row in ds:
        test_code = "\n".join(row.get("test_list", []))
        tasks.append({
            "task_id": f"mbpp_{row['task_id']}",
            "title": row["prompt"][:60],
            "target_category": "logical",
            "problem_type": "mbpp",
            "requirement": row["prompt"],
            "test_input": "",
            "expected_output": "",
            "test_cases": test_code,
            "relevant_doc_keywords": [],
        })
    return tasks


def load_humaneval_benchmark(limit: int = 20) -> list[dict[str, Any]]:
    """Load HumanEval tasks via HuggingFace datasets."""
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split=f"test[:{limit}]")
    tasks: list[dict[str, Any]] = []
    for row in ds:
        entry = row["entry_point"]
        test_code = f"{row['test']}\ncheck({entry})"
        tasks.append({
            "task_id": row["task_id"].replace("/", "_"),
            "title": f"HumanEval {entry}",
            "target_category": "logical",
            "problem_type": "humaneval",
            "requirement": (
                f"Complete the following Python function `{entry}`:\n\n{row['prompt']}"
            ),
            "test_input": "",
            "expected_output": "",
            "test_cases": test_code,
            "relevant_doc_keywords": [],
        })
    return tasks


# ---------------------------------------------------------------------------
# Retrieval Quality Metrics (Report Section 8.4)
# ---------------------------------------------------------------------------

def compute_retrieval_quality(
    run_history: RunHistory,
    relevant_keywords: list[str],
) -> tuple[float | None, float | None]:
    """Compute approximate Retrieval Precision and Recall from retrieved chunks."""
    triggered = [r for r in run_history.retrievals if r.triggered and r.chunks]
    if not triggered or not relevant_keywords:
        return None, None

    kw_lower = [k.lower() for k in relevant_keywords]
    total_chunks = 0
    relevant_chunks = 0
    matched_keywords: set[str] = set()

    for rec in triggered:
        for ch in rec.chunks:
            total_chunks += 1
            text = (ch.get("content", "") + " " + json.dumps(ch.get("metadata", {}))).lower()
            hits = [k for k in kw_lower if k in text]
            if hits:
                relevant_chunks += 1
                matched_keywords.update(hits)

    precision = relevant_chunks / total_chunks if total_chunks else 0.0
    recall = len(matched_keywords) / len(kw_lower) if kw_lower else 0.0
    return precision, recall


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_suite(
    tasks: list[dict[str, Any]],
    configs: list[str],
    max_iterations: int = 3,
    execution_timeout: int = 5,
    output_prefix: str = "experiment_runs",
) -> pd.DataFrame:
    """Execute every task across every chosen configuration and save CSV/JSON."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Always disable run_memory during controlled experiments to prevent cross-task leakage
    backend = OdinBackend(use_run_memory=False)
    rows: list[dict[str, Any]] = []

    for cfg_name in configs:
        mode = CONFIG_MAP[cfg_name]
        logger.info("=== Running %s (mode=%s) over %d tasks ===", cfg_name, mode, len(tasks))

        for task in tasks:
            logger.info("[%s] Task %s (%s)", cfg_name, task["task_id"], task["target_category"])
            rh = await backend.run_pipeline(
                requirement=task["requirement"],
                test_input=task.get("test_input", ""),
                test_cases=task.get("test_cases", ""),
                expected_output=task.get("expected_output", ""),
                experiment_mode=mode,
                max_iterations=max_iterations,
                execution_timeout=execution_timeout,
                use_run_memory=False,
            )
            metrics = backend.compute_metrics(str(rh.run_id))
            precision, recall = compute_retrieval_quality(
                rh, task.get("relevant_doc_keywords", [])
            )
            detected_categories = [ea.category for ea in rh.error_analyses]
            total_chunks = sum(len(r.chunks) for r in rh.retrievals if r.triggered)

            rows.append({
                "timestamp": datetime.now().isoformat(),
                "model": config.gemini_model,
                "config": cfg_name,
                "experiment_mode": mode,
                "task_id": task["task_id"],
                "title": task.get("title", ""),
                "target_category": task["target_category"],
                "problem_type": task.get("problem_type", "custom"),
                "run_id": str(rh.run_id),
                "final_status": rh.final_status,
                "success": bool(metrics["success"]),
                "first_attempt_success": bool(metrics["success"] and metrics["total_iterations"] == 1),
                "partial_repair": bool(metrics["partial_repair"]),
                "tests_passed": metrics["tests_passed"],
                "tests_total": metrics["tests_total"],
                "test_pass_rate": metrics["test_pass_rate"],
                "total_iterations": metrics["total_iterations"],
                "wall_time_seconds": metrics["total_wall_time_seconds"],
                "total_execution_calls": metrics["total_execution_calls"],
                "total_tool_calls": metrics["total_execution_calls"] + metrics["total_retrieval_count"],
                "total_input_tokens": metrics["total_input_tokens"],
                "total_output_tokens": metrics["total_output_tokens"],
                "total_llm_tokens": metrics["total_llm_tokens"],
                "retrieval_decisions": metrics["total_retrieval_decisions"],
                "retrieval_triggered_count": metrics["total_retrieval_count"],
                "retrieved_chunks_count": total_chunks,
                "retrieved_context_tokens": metrics["total_retrieval_tokens"],
                "retrieval_latency_ms": metrics["total_retrieval_latency_ms"],
                "retrieval_precision": precision,
                "retrieval_recall": recall,
                "detected_categories": ",".join(detected_categories),
                "error_message": rh.error_message or "",
            })

    df = pd.DataFrame(rows)
    csv_path = _RESULTS_DIR / f"{output_prefix}.csv"
    json_path = _RESULTS_DIR / f"{output_prefix}.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logger.info("Saved %d experiment records to %s", len(df), csv_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ODIN comparative experiments (Configs A/B/C/D)")
    parser.add_argument(
        "--dataset",
        choices=["custom", "mbpp", "humaneval"],
        default="custom",
        help="Benchmark dataset to evaluate",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIG_MAP.keys()),
        default=list(CONFIG_MAP.keys()),
        help="Which experimental configurations to run",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tasks (0 = all)")
    parser.add_argument(
        "--per-category",
        type=int,
        default=0,
        help="Sample N tasks per error category from custom benchmark (0 = disabled)",
    )
    parser.add_argument("--max-iterations", type=int, default=3, help="Max iterations for iterative configs")
    parser.add_argument("--timeout", type=int, default=10, help="Per-execution sandbox timeout (sec)")
    parser.add_argument("--output-prefix", default="experiment_runs", help="Output filename prefix")
    args = parser.parse_args()

    if args.dataset == "custom":
        tasks = load_custom_benchmark()
        if args.per_category > 0:
            by_cat: dict[str, list[dict[str, Any]]] = {}
            for t in tasks:
                cat = t["target_category"]
                if len(by_cat.get(cat, [])) < args.per_category:
                    by_cat.setdefault(cat, []).append(t)
            tasks = [t for group in by_cat.values() for t in group]
        elif args.limit > 0:
            tasks = tasks[:args.limit]
    elif args.dataset == "mbpp":
        tasks = load_mbpp_benchmark(limit=args.limit or 20)
    else:
        tasks = load_humaneval_benchmark(limit=args.limit or 20)

    asyncio.run(
        run_suite(
            tasks=tasks,
            configs=args.configs,
            max_iterations=args.max_iterations,
            execution_timeout=args.timeout,
            output_prefix=args.output_prefix,
        )
    )


if __name__ == "__main__":
    main()
