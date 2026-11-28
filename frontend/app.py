"""ODIN Streamlit Web Interface.

Provides:
1. Live Self-Debugger: run tasks with stdin, expected output, and Python assert
   test suites across all 4 experimental configurations (Direct, No-RAG,
   Adaptive RAG, Always-On RAG), inspect every iteration + RAG chunks, send
   feedback, download code, or save to a local folder.
2. Run History & Metrics: inspect every past run saved in outputs/ with Pandas
   summaries and category distributions.
3. Research Evaluation Dashboard: visualize comparative experiment CSV results
   (Config A / B / C / Always-On RAG) answering RQ1–RQ6.

Launch with:
    streamlit run frontend/app.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure src/ is importable when invoked directly via `streamlit run frontend/app.py`
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from odin.backend import OdinBackend
from odin.config import config
from odin.persistence import RunHistoryPersistence, _make_run_dir_name
from odin.state import ExperimentMode, RunHistory

# ---------------------------------------------------------------------------
# Presets covering all problem & error categories
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, str]] = {
    "Custom task": {
        "requirement": "",
        "test_input": "",
        "expected_output": "",
        "test_cases": "",
    },
    "DSA — Minimum Subarray Partition Cost (Report Demo)": {
        "requirement": (
            "Write a Python solution for this problem:\n"
            "Find the minimum cost to divide an array of n positive integers into exactly k "
            "non-empty contiguous subarrays. The cost of each subarray is the square of its sum. "
            "Use dynamic programming with O(k * n^2) time complexity. Read input from standard "
            "input (n and k on line 1, array elements on line 2) and print only the final answer."
        ),
        "test_input": "5 2\n1 2 3 4 5\n",
        "expected_output": "117",
        "test_cases": "",
    },
    "Algorithm + Unit Tests — Second Largest Distinct": {
        "requirement": (
            "Write a Python function `second_largest(nums)` that returns the second largest "
            "distinct integer in a list of integers, or None if fewer than 2 distinct values exist."
        ),
        "test_input": "",
        "expected_output": "",
        "test_cases": (
            "assert second_largest([3, 1, 4, 4, 2]) == 3\n"
            "assert second_largest([5, 5, 5]) is None\n"
            "assert second_largest([-1, -5, -2]) == -2\n"
            "assert second_largest([]) is None"
        ),
    },
    "Version-Specific RAG — Pandas 2.x/3.x DataFrame.append": {
        "requirement": (
            "Write a Python function `append_row(df, row_dict)` using pandas that returns a new "
            "DataFrame with `row_dict` appended as the last row. Initially attempt df.append(row_dict, ignore_index=True) "
            "and repair if deprecated/removed."
        ),
        "test_input": "",
        "expected_output": "",
        "test_cases": (
            "import pandas as pd\n"
            "df = pd.DataFrame([{'a': 1, 'b': 2}])\n"
            "res = append_row(df, {'a': 3, 'b': 4})\n"
            "assert len(res) == 2\n"
            "assert list(res['a']) == [1, 3]"
        ),
    },
    "Uncommon Pattern RAG — itertools.groupby Word Counts": {
        "requirement": (
            "Write a Python function `count_words_groupby(words)` using `itertools.groupby` that "
            "returns a dict mapping each word to its total count even when identical words are not adjacent."
        ),
        "test_input": "",
        "expected_output": "",
        "test_cases": (
            "assert count_words_groupby(['apple', 'banana', 'apple', 'cherry']) == {'apple': 2, 'banana': 1, 'cherry': 1}\n"
            "assert count_words_groupby([]) == {}"
        ),
    },
}

MODE_LABELS: dict[str, ExperimentMode] = {
    "Config C — Adaptive RAG (Recommended)": "adaptive_rag",
    "Config A — Direct Generation (1-shot, no repair)": "direct",
    "Config B — Iterative Self-Debugging (No RAG)": "no_rag",
    "Config D — Always-On RAG (Retrieve on every error)": "always_rag",
}


# ---------------------------------------------------------------------------
# Cached backend helper
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Initializing ODIN backend...")
def _get_backend(use_run_memory: bool) -> OdinBackend:
    return OdinBackend(use_run_memory=use_run_memory)


def _run_async(coro):
    return asyncio.run(coro)


def _save_to_user_folder(run_history: RunHistory, folder_path: str) -> str:
    base = Path(folder_path).expanduser().resolve()
    if base.exists() and base.is_file():
        raise ValueError(f"'{base}' is a file, not a folder. Please enter a folder path.")
    dest_dir = base / _make_run_dir_name(run_history.run_id, run_history.requirement)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in run_history.code_versions[-1].files:
        target = dest_dir / f.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f.content, encoding="utf-8")
    return str(dest_dir)


def _render_run_details(
    run_history: RunHistory,
    persistence: RunHistoryPersistence,
    key_prefix: str = "live",
) -> None:
    """Render full per-iteration breakdown for a RunHistory object."""
    status = run_history.final_status
    if status == "success":
        st.success(f"Succeeded in {run_history.total_iterations} iteration(s)  |  Mode: `{run_history.experiment_mode}`")
    elif status == "max_iterations_reached":
        st.warning(f"Reached max iterations ({run_history.total_iterations}) without passing all checks.")
    else:
        st.error(f"Run failed: {run_history.error_message or 'Unknown error'}")

    # Top-level metrics strip
    try:
        metrics = persistence.compute_metrics(str(run_history.run_id))
    except FileNotFoundError:
        metrics = None

    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Iterations", metrics["total_iterations"])
        c2.metric("LLM Tokens", metrics["total_llm_tokens"])
        c3.metric("RAG Retrievals", f"{metrics['total_retrieval_count']}/{metrics['total_retrieval_decisions']}")
        c4.metric("Wall Time", f"{metrics['total_wall_time_seconds']:.2f}s")
        if metrics["tests_total"] > 0:
            c5.metric("Tests Passed", f"{metrics['tests_passed']}/{metrics['tests_total']}")
        else:
            c5.metric("Exec Calls", metrics["total_execution_calls"])

    # Index records by iteration
    exec_by_iter = {e.iteration: e for e in run_history.executions}
    err_by_iter = {a.iteration: a for a in run_history.error_analyses}
    ret_by_iter = {r.iteration: r for r in run_history.retrievals}
    tok_by_iter = {t.iteration: t for t in run_history.token_usage}

    for cv in run_history.code_versions:
        it = cv.iteration
        ex = exec_by_iter.get(it)
        ea = err_by_iter.get(it)
        rt = ret_by_iter.get(it)
        tk = tok_by_iter.get(it)

        badge = "PASS" if (ex and ex.success) else "FAIL"
        header = f"Iteration {it} — {badge}"
        if ea:
            header += f" | Error: {ea.category}"
        if rt:
            header += f" | RAG: {'TRIGGERED' if rt.triggered else 'SKIPPED'}"

        with st.expander(header, expanded=(it == run_history.code_versions[-1].iteration)):
            for f in cv.files:
                st.caption(f"File: `{f.path}`")
                st.code(f.content, language="python")

            if ex:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Execution**: exit `{ex.exit_code}` in `{ex.execution_time_ms:.1f} ms`")
                    if ex.tests_total > 0:
                        st.markdown(f"**Tests**: `{ex.tests_passed}/{ex.tests_total}` passed")
                    if ex.stdout:
                        st.text_area("STDOUT", ex.stdout, height=90, disabled=True, key=f"{key_prefix}_stdout_{run_history.run_id}_{it}")
                with col_b:
                    if tk:
                        st.markdown(
                            f"**Tokens (`{tk.call_type}`)**: {tk.total_tokens} "
                            f"(in: {tk.input_tokens}, out: {tk.output_tokens})"
                        )
                    if ex.stderr:
                        st.text_area("STDERR / Traceback", ex.stderr, height=90, disabled=True, key=f"{key_prefix}_stderr_{run_history.run_id}_{it}")

            if ea:
                st.info(f"**Error Analysis** — Category: `{ea.category}` | {ea.reasoning}")

            if rt:
                if rt.triggered:
                    st.warning(
                        f"**RAG Retrieval TRIGGERED** ({len(rt.chunks)} chunks, "
                        f"{rt.retrieval_latency_ms:.1f} ms, ~{rt.total_tokens} ctx tokens) — {rt.decision_reason}"
                    )
                    for idx, ch in enumerate(rt.chunks, 1):
                        sim = ch.get("similarity_score", 0.0)
                        meta = ch.get("metadata", {})
                        st.caption(f"Chunk {idx} (similarity={sim:.2f}, library={meta.get('library', 'python')}, type={meta.get('doc_type', 'doc')})")
                        st.code(ch.get("content", ""), language="markdown")
                else:
                    st.caption(f"RAG Retrieval SKIPPED — {rt.decision_reason}")

    # Download final code
    if run_history.code_versions:
        final_files = run_history.code_versions[-1].files
        if len(final_files) == 1:
            st.download_button(
                label=f"Download {final_files[0].path}",
                data=final_files[0].content,
                file_name=final_files[0].path,
                mime="text/x-python",
                key=f"{key_prefix}_dl_{run_history.run_id}",
            )


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ODIN — Adaptive RAG Code Repair", page_icon="O", layout="wide")
st.title("ODIN — Orchestrated Debugging & Intelligent Navigation")
st.caption("Agentic Self-Debugging Framework for Automated Code Repair using Adaptive Retrieval-Augmented Generation")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    mode_label = st.selectbox("Experiment Mode", list(MODE_LABELS.keys()), index=0)
    experiment_mode: ExperimentMode = MODE_LABELS[mode_label]

    max_iterations = st.slider(
        "Max Iterations",
        min_value=1,
        max_value=5,
        value=1 if experiment_mode == "direct" else min(config.max_iterations, 5),
        disabled=(experiment_mode == "direct"),
    )
    execution_timeout = st.slider("Execution Timeout (sec)", min_value=1, max_value=20, value=config.execution_timeout)
    use_run_memory = st.checkbox(
        "Use Past-Run Memory",
        value=False,
        help="Disable for isolated controlled experiments; enable to reuse similar past solutions.",
    )
    output_dir = st.text_input(
        "Save working code to folder (optional)",
        value="",
        placeholder=r"e.g. C:\Users\khush\OneDrive\Desktop\odin_outputs",
    )

    st.divider()
    st.markdown(f"**Model:** `{config.gemini_model}`")
    st.markdown(f"**Sandbox:** `{config.sandbox_mode}` (isolated temp dir)")
    st.markdown(f"**API Key:** {'Configured' if config.gemini_api_key else 'Missing in .env'}")

persistence = RunHistoryPersistence()

tab_debug, tab_history, tab_research = st.tabs([
    "Live Self-Debugger",
    "Run History & Metrics",
    "Research Evaluation (RQ1–RQ6)",
])

# ---------------------------------------------------------------------------
# Tab 1: Live Self-Debugger
# ---------------------------------------------------------------------------
with tab_debug:
    preset_name = st.selectbox("Load Example Task", list(PRESETS.keys()))
    preset = PRESETS[preset_name]

    requirement = st.text_area(
        "Programming Task (Natural Language)",
        value=preset["requirement"],
        height=110,
        placeholder="Describe the Python program or function to generate and debug...",
    )

    col_in, col_exp = st.columns(2)
    with col_in:
        test_input = st.text_area(
            "Test Input (stdin, optional)",
            value=preset["test_input"],
            height=90,
            placeholder="Input piped to stdin during execution",
        )
    with col_exp:
        expected_output = st.text_area(
            "Expected Output (stdout, optional)",
            value=preset["expected_output"],
            height=90,
            placeholder="If provided, stdout must match this output to pass",
        )

    test_cases = st.text_area(
        "Test Cases (Python assert statements, optional)",
        value=preset["test_cases"],
        height=100,
        placeholder="e.g.\nassert solve([1, 2, 3]) == 6\nassert solve([]) == 0",
    )

    run_clicked = st.button("Run ODIN Pipeline", type="primary", disabled=not requirement.strip())

    if run_clicked:
        if not config.gemini_api_key:
            st.error("GEMINI_API_KEY is missing. Please add it to your `.env` file.")
        else:
            with st.spinner(f"Running ODIN ({experiment_mode})..."):
                backend = _get_backend(use_run_memory)
                run_history = _run_async(
                    backend.run_pipeline(
                        requirement=requirement,
                        test_input=test_input,
                        test_cases=test_cases,
                        expected_output=expected_output,
                        experiment_mode=experiment_mode,
                        max_iterations=max_iterations,
                        execution_timeout=execution_timeout,
                        use_run_memory=use_run_memory,
                    )
                )
            st.session_state["active_run_id"] = str(run_history.run_id)
            st.session_state["active_run"] = run_history

            if output_dir.strip() and run_history.final_status == "success":
                try:
                    saved = _save_to_user_folder(run_history, output_dir.strip())
                    st.toast(f"Saved working code to {saved}")
                    st.session_state["last_saved_dir"] = saved
                except Exception as exc:
                    st.error(f"Could not save to output folder: {exc}")

    active_run: RunHistory | None = st.session_state.get("active_run")
    if active_run is not None:
        if st.session_state.get("last_saved_dir"):
            st.info(f"Final working code copied to: `{st.session_state['last_saved_dir']}`")
        _render_run_details(active_run, persistence, key_prefix="live")

        st.divider()
        st.subheader("Send Feedback to Revise This Run")
        with st.form("feedback_form", clear_on_submit=True):
            fb_text = st.text_input(
                "Describe what to change or fix",
                placeholder="e.g. 'Read input at runtime and handle empty input gracefully'",
                key="feedback_box",
            )
            fb_submitted = st.form_submit_button("Send Feedback")

        if fb_submitted:
            if not fb_text.strip():
                st.warning("Please enter feedback before submitting.")
            else:
                with st.spinner("Applying feedback and re-verifying in sandbox..."):
                    backend = _get_backend(use_run_memory)
                    updated = _run_async(
                        backend.continue_run(
                            run_history=active_run,
                            user_feedback=fb_text.strip(),
                            test_input=test_input,
                            test_cases=test_cases,
                            expected_output=expected_output,
                            execution_timeout=execution_timeout,
                        )
                    )
                st.session_state["active_run"] = updated
                if output_dir.strip() and updated.final_status == "success":
                    try:
                        st.session_state["last_saved_dir"] = _save_to_user_folder(updated, output_dir.strip())
                    except Exception as exc:
                        st.error(f"Could not save to output folder: {exc}")
                st.rerun()

# ---------------------------------------------------------------------------
# Tab 2: Run History & Metrics
# ---------------------------------------------------------------------------
with tab_history:
    runs = persistence.list_runs()
    if not runs:
        st.info("No runs saved in `outputs/` yet. Run a task in the Live Self-Debugger tab first.")
    else:
        rows = []
        for r in runs:
            try:
                m = persistence.compute_metrics(r["run_id"])
                rows.append({
                    "Run ID": r["run_id"][:8],
                    "Mode": m["experiment_mode"],
                    "Status": m["final_status"],
                    "Iters": m["total_iterations"],
                    "Tokens": m["total_llm_tokens"],
                    "RAG Calls": m["total_retrieval_count"],
                    "Wall (s)": round(m["total_wall_time_seconds"], 2),
                    "Task": r["requirement"],
                })
            except Exception:
                continue

        if rows:
            df_runs = pd.DataFrame(rows)
            st.dataframe(df_runs, width="stretch", hide_index=True)

            batch = persistence.compute_batch_metrics([r["run_id"] for r in runs])
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Total Runs", batch["total_runs"])
            b2.metric("Repair Success Rate", f"{batch['success_rate']:.1%}")
            b3.metric("Avg Iterations", f"{batch['avg_iterations']:.2f}")
            b4.metric("Retrieval Frequency", f"{batch['retrieval_frequency']:.1%}")

            if batch["error_category_distribution"]:
                st.subheader("Error Category Distribution")
                cat_df = pd.DataFrame(
                    list(batch["error_category_distribution"].items()),
                    columns=["Error Category", "Count"],
                ).set_index("Error Category")
                st.bar_chart(cat_df)

        selected_id = st.selectbox(
            "Inspect a saved run",
            options=[r["run_id"] for r in runs],
            format_func=lambda rid: next(
                f"{r['run_id'][:8]} | {r['final_status']} | {r['requirement'][:70]}"
                for r in runs if r["run_id"] == rid
            ),
        )
        if selected_id:
            loaded = persistence.load_run(selected_id)
            _render_run_details(loaded, persistence, key_prefix="history")

# ---------------------------------------------------------------------------
# Tab 3: Research Evaluation Dashboard
# ---------------------------------------------------------------------------
with tab_research:
    results_dir = _ROOT / "experiments" / "results"
    csv_files = sorted(results_dir.glob("*.csv"), reverse=True) if results_dir.exists() else []

    st.markdown(
        "Run controlled comparative experiments across **Config A (Direct)**, **Config B (No-RAG)**, "
        "**Config C (Adaptive RAG)**, and **Config D (Always-On RAG)** using:\n"
        "```powershell\n"
        ".\\.venv\\Scripts\\python.exe -m experiments.run_experiments --limit 6\n"
        ".\\.venv\\Scripts\\python.exe -m experiments.analyze_results\n"
        "```"
    )

    if not csv_files:
        st.info("No experiment CSVs found in `experiments/results/` yet. Run the experiment harness above to populate this dashboard.")
    else:
        chosen_csv = st.selectbox("Select experiment result file", csv_files, format_func=lambda p: p.name)
        df = pd.read_csv(chosen_csv)
        st.dataframe(df, width="stretch", hide_index=True)

        if {"config", "success", "total_iterations", "total_llm_tokens"}.issubset(df.columns):
            st.subheader("Comparative Summary (Direct → No-RAG → Adaptive RAG → Always-On RAG)")
            summary = df.groupby("config", as_index=False).agg(
                problems=("task_id", "count"),
                repair_success_rate=("success", "mean"),
                test_pass_rate=("test_pass_rate", "mean"),
                first_attempt_rate=("first_attempt_success", "mean"),
                mean_iterations=("total_iterations", "mean"),
                median_iterations=("total_iterations", "median"),
                avg_tokens=("total_llm_tokens", "mean"),
                retrieval_freq=("retrieval_triggered_count", lambda s: (s > 0).mean()),
                avg_wall_sec=("wall_time_seconds", "mean"),
            )
            st.dataframe(summary, width="stretch", hide_index=True)

            c_left, c_right = st.columns(2)
            with c_left:
                st.caption("Repair Success Rate by Configuration")
                st.bar_chart(summary.set_index("config")["repair_success_rate"])
            with c_right:
                st.caption("Average LLM Tokens by Configuration")
                st.bar_chart(summary.set_index("config")["avg_tokens"])

        if {"target_category", "config", "success"}.issubset(df.columns):
            st.subheader("Error-Category-Wise Repair Success Rate (RQ3)")
            pivot = df.pivot_table(
                index="target_category",
                columns="config",
                values="success",
                aggfunc="mean",
            ).fillna(0.0)
            st.dataframe(pivot, width="stretch")
            st.bar_chart(pivot)

        png_files = sorted(results_dir.glob("*.png"))
        if png_files:
            st.subheader("Generated Research Plots")
            for png in png_files:
                st.image(str(png), caption=png.name)
