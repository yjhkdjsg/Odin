"""Pandas + Matplotlib evaluation script for ODIN experiments (Report Section 8).

Reads an experiment CSV produced by `experiments.run_experiments` and outputs:
1. `summary_by_config.csv`   — Config A / B / C / D comparison (effectiveness,
                               efficiency, retrieval frequency, token cost, precision/recall)
2. `summary_by_category.csv` — Error-category-wise breakdown across all 6 categories (RQ3)
3. `negative_effects.csv`    — Cases where RAG failed or regressed vs No-RAG,
                               or triggered unnecessary retrieval (Section 9.6)
4. Publication-ready PNG charts in `experiments/results/`:
   - `success_rate_by_config.png`
   - `tokens_and_latency_by_config.png`
   - `category_comparison.png`

Usage:
    python -m experiments.analyze_results [--input experiments/results/experiment_runs.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import pandas as pd

_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def analyze_dataframe(df: pd.DataFrame, out_dir: Path = _RESULTS_DIR) -> dict[str, pd.DataFrame]:
    """Compute all Section 8 summary tables and save CSVs + PNG charts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall configuration comparison (Sections 8.1, 8.2, 8.3, 8.4, 8.6)
    summary_by_config = df.groupby("config", as_index=False).agg(
        total_problems=("task_id", "count"),
        repair_success_rate=("success", "mean"),
        test_pass_rate=("test_pass_rate", "mean"),
        first_attempt_success_rate=("first_attempt_success", "mean"),
        partial_repair_rate=("partial_repair", "mean"),
        mean_iterations=("total_iterations", "mean"),
        median_iterations=("total_iterations", "median"),
        mean_wall_time_sec=("wall_time_seconds", "mean"),
        mean_tool_calls=("total_tool_calls", "mean"),
        total_retrieval_calls=("retrieval_triggered_count", "sum"),
        total_retrieval_decisions=("retrieval_decisions", "sum"),
        mean_input_tokens=("total_input_tokens", "mean"),
        mean_output_tokens=("total_output_tokens", "mean"),
        mean_total_tokens=("total_llm_tokens", "mean"),
        mean_retrieved_ctx_tokens=("retrieved_context_tokens", "mean"),
        mean_retrieval_latency_ms=("retrieval_latency_ms", "mean"),
        mean_retrieval_precision=("retrieval_precision", "mean"),
        mean_retrieval_recall=("retrieval_recall", "mean"),
    )
    summary_by_config["retrieval_frequency"] = (
        summary_by_config["total_retrieval_calls"]
        / summary_by_config["total_retrieval_decisions"].replace(0, pd.NA)
    ).fillna(0.0)
    summary_by_config.to_csv(out_dir / "summary_by_config.csv", index=False)

    # 2. Error-category-wise breakdown (Section 8.5)
    summary_by_category = df.groupby(["target_category", "config"], as_index=False).agg(
        problems=("task_id", "count"),
        repair_success_rate=("success", "mean"),
        test_pass_rate=("test_pass_rate", "mean"),
        mean_iterations=("total_iterations", "mean"),
        mean_tokens=("total_llm_tokens", "mean"),
        retrieval_calls=("retrieval_triggered_count", "sum"),
    )
    summary_by_category.to_csv(out_dir / "summary_by_category.csv", index=False)

    # 3. Negative effects & unnecessary retrieval analysis (Section 9.6)
    neg_rows = []
    if "config_b_no_rag" in set(df["config"]):
        base = df[df["config"] == "config_b_no_rag"].set_index("task_id")
        for rag_cfg in ("config_c_adaptive_rag", "config_d_always_rag"):
            sub = df[df["config"] == rag_cfg]
            for _, row in sub.iterrows():
                tid = row["task_id"]
                if tid not in base.index:
                    continue
                b = base.loc[tid]
                regressed = bool(b["success"]) and not bool(row["success"])
                unnecessary = (
                    row["retrieval_triggered_count"] > 0
                    and bool(b["success"])
                    and bool(row["success"])
                    and row["total_iterations"] >= b["total_iterations"]
                )
                if regressed or unnecessary:
                    neg_rows.append({
                        "task_id": tid,
                        "target_category": row["target_category"],
                        "rag_config": rag_cfg,
                        "issue_type": "regression" if regressed else "unnecessary_retrieval",
                        "no_rag_iters": b["total_iterations"],
                        "rag_iters": row["total_iterations"],
                        "extra_tokens": row["total_llm_tokens"] - b["total_llm_tokens"],
                        "retrieval_calls": row["retrieval_triggered_count"],
                    })
    negative_df = pd.DataFrame(neg_rows)
    negative_df.to_csv(out_dir / "negative_effects.csv", index=False)

    # 4. Charts
    _plot_success_by_config(summary_by_config, out_dir / "success_rate_by_config.png")
    _plot_tokens_and_time(summary_by_config, out_dir / "tokens_and_latency_by_config.png")
    _plot_category_comparison(summary_by_category, out_dir / "category_comparison.png")

    return {
        "summary_by_config": summary_by_config,
        "summary_by_category": summary_by_category,
        "negative_effects": negative_df,
    }


def _plot_success_by_config(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(summary))
    width = 0.35
    ax.bar([i - width / 2 for i in x], summary["repair_success_rate"] * 100, width, label="Repair Success Rate (%)")
    ax.bar([i + width / 2 for i in x], summary["test_pass_rate"] * 100, width, label="Test Pass Rate (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["config"], rotation=15, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 105)
    ax.set_title("ODIN Repair Effectiveness Across Configurations")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_tokens_and_time(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.bar(summary["config"], summary["mean_total_tokens"], color="#3b82f6")
    ax1.set_title("Mean LLM Tokens per Task")
    ax1.set_ylabel("Tokens")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2.bar(summary["config"], summary["retrieval_frequency"] * 100, color="#f59e0b")
    ax2.set_title("Retrieval Frequency (%)")
    ax2.set_ylabel("Percentage of Debug Attempts")
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_category_comparison(cat_summary: pd.DataFrame, path: Path) -> None:
    if cat_summary.empty:
        return
    pivot = cat_summary.pivot_table(
        index="target_category",
        columns="config",
        values="repair_success_rate",
        aggfunc="mean",
    ).fillna(0.0) * 100
    ax = pivot.plot(kind="bar", figsize=(9, 4.8))
    ax.set_ylabel("Repair Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Repair Success Rate by Error Category & Configuration (RQ3)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ODIN experiment results with Pandas")
    parser.add_argument(
        "--input",
        type=Path,
        default=_RESULTS_DIR / "experiment_runs.csv",
        help="Path to experiment_runs.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_RESULTS_DIR,
        help="Directory for output summary CSVs and PNG plots",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Experiment CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    results = analyze_dataframe(df, out_dir=args.out_dir)

    print("\n=== Comparative Summary by Configuration ===")
    print(results["summary_by_config"].to_string(index=False))
    print("\n=== Error-Category-Wise Breakdown ===")
    print(results["summary_by_category"].to_string(index=False))
    print(f"\nSaved summaries and PNG plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
