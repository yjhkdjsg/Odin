# ODIN — Orchestrated Debugging and Intelligent Navigation

**Agentic Self-Debugging Framework for Automated Code Repair using Adaptive Retrieval-Augmented Generation (RAG)**

ODIN iteratively generates, executes in an isolated sandbox, verifies test cases, analyzes failures, conditionally retrieves external documentation via ChromaDB, and repairs Python code—while recording complete per-iteration telemetry for comparative research experiments (**Configs A / B / C / Always-On RAG**).

---

## Core Architecture & Features

```text
User / Benchmark Task
        │
        ▼
  generate_node (Gemini via google-genai)
        │
        ▼
  execute_node (Isolated temp-dir sandbox + OS limits / optional Docker + test-case & stdout verifier)
        │
        ▼
  verify_node ──► Pass OR iteration >= max_iterations ──► Persist to outputs/ ──► END
        │
     [Fail]
        ▼
  analyze_error_node (6 categories: syntax, runtime, logical, library_api, version_specific, uncommon_pattern)
        │
        ▼
  retrieve_node (Adaptive RAG gate: skips syntax/runtime/logical; queries ChromaDB for library_api/version_specific/uncommon_pattern)
        │
        ▼
  repair_node (failing code + traceback + prior debugging history + optional RAG chunks) ──► loops to execute_node
```

- **Two User Interfaces + REST API**:
  - **Streamlit Website** (`frontend/app.py`): Live self-debugger with presets, per-iteration code/RAG transparency, feedback revisions, code downloads, Run History browser, and Research Evaluation dashboard.
  - **FastAPI Backend** (`src/odin/api.py`): REST endpoints (`/health`, `/runs`, `/runs/{id}/feedback`, `/runs/{id}`).
  - **Textual Terminal UI** (`src/odin/tui.py`): Interactive terminal interface (`python -m odin`).
- **Hardened Sandbox Execution** (`src/odin/executor.py`):
  - Runs **both** single-file and multi-file code inside an isolated temporary directory (`cwd=tmp_dir`).
  - Strips secret environment variables (`GEMINI_API_KEY`, tokens, etc.) from child processes.
  - Enforces wall-clock timeout, Unix `resource.setrlimit`, Windows Win32 Job Object memory/process caps, and optional Docker container isolation (`ODIN_SANDBOX_MODE=docker`, `--network none`).
  - Functional verification via Python `assert` test suites (`test_cases`) and/or `expected_output` comparison (computes **Test Pass Rate** & **Partial Repair Rate**).
- **Research Experiment Harness** (`experiments/`):
  - Curated 18-task 6-category benchmark (`experiments/benchmark_suite.json`) plus HuggingFace loaders for **MBPP** and **HumanEval**.
  - Controlled comparative runner (`experiments/run_experiments.py`) across **Config A (Direct)**, **Config B (No-RAG)**, **Config C (Adaptive RAG)**, and **Config D (Always-On RAG)** with `run_memory` disabled.
  - Pandas + Matplotlib analyzer (`experiments/analyze_results.py`) producing Section 8 metrics tables, negative-effect analysis, and PNG plots.

---

## Quick Setup (Windows PowerShell)

```powershell
git clone https://github.com/kxi2205/Odin-Framework.git
cd Odin-Framework
uv sync
Copy-Item .env.example .env
```

Edit `.env` and add your Gemini API key:
```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-flash-lite-latest
ODIN_MAX_ITERATIONS=3
ODIN_EXECUTION_TIMEOUT=5
ODIN_SANDBOX_MODE=subprocess
```

Ingest the knowledge base into ChromaDB (run once, and again whenever you add docs to `knowledge/`):
```powershell
.\.venv\Scripts\python.exe -m odin.rag.ingest --reset
```

---

## Running ODIN

### 1. Streamlit Website (Recommended)
```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```
Opens in your browser at `http://localhost:8501`.

### 2. FastAPI REST Backend (Optional)
```powershell
.\.venv\Scripts\python.exe -m odin.api
```
Interactive API docs at `http://127.0.0.1:8000/docs`.

### 3. Textual Terminal UI
```powershell
.\.venv\Scripts\python.exe -m odin
```

---

## Running Comparative Experiments (RQ1–RQ6)

```powershell
# Run all 4 configurations (Direct, No-RAG, Adaptive RAG, Always-On RAG) on the 6-category benchmark
.\.venv\Scripts\python.exe -m experiments.run_experiments --dataset custom --max-iterations 3

# Or run on MBPP / HumanEval subsets
.\.venv\Scripts\python.exe -m experiments.run_experiments --dataset mbpp --limit 20
.\.venv\Scripts\python.exe -m experiments.run_experiments --dataset humaneval --limit 20

# Generate Pandas summary tables & PNG charts in experiments/results/
.\.venv\Scripts\python.exe -m experiments.analyze_results
```

---

## Running Unit Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

---

## Project Structure

```text
frontend/
└── app.py                     # Streamlit website (Debugger, Run History, Research Dashboard)
src/odin/
├── api.py                     # FastAPI REST backend
├── backend.py                 # UI-agnostic async pipeline facade
├── config.py                  # Pydantic config (env vars, sandbox, model, paths)
├── error_classifier.py        # 6-category heuristic traceback classifier
├── executor.py                # Isolated sandbox + test-case & expected-output verifier
├── graph.py                   # LangGraph workflow definition
├── llm.py                     # google-genai client with retry, history & token tracking
├── nodes.py                   # Graph nodes (generate, feedback, execute, verify, analyze, retrieve, repair)
├── persistence.py             # Per-run folder persistence & batch metrics
├── routing.py                 # Deterministic Adaptive RAG routing table
├── state.py                   # Pydantic state models & RunHistory
├── tui.py                     # Textual terminal UI
└── rag/
    ├── ingest.py              # ChromaDB knowledge ingestion CLI
    ├── retriever.py           # Adaptive RAG retriever (all-MiniLM-L6-v2)
    └── run_memory.py          # Optional cross-run semantic memory
knowledge/
├── examples.md                # 20 curated error-fix pairs
└── python_libraries_guide.md  # Pandas/NumPy/Sklearn/Pydantic/Stdlib API & version migration guide
experiments/
├── benchmark_suite.json       # 18-task 6-category benchmark (DSA, algorithms, API, version, uncommon)
├── run_experiments.py         # Config A/B/C/D comparative runner (Custom / MBPP / HumanEval)
└── analyze_results.py         # Pandas & Matplotlib Section 8 evaluation analyzer
tests/                         # Full pytest unit test suite
```
