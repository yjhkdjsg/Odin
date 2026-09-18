# ODIN

### Orchestrated Debugging and Intelligent Navigation

ODIN is an agentic AI framework for **automated code debugging and repair** using Large Language Models and Retrieval-Augmented Generation (RAG).

It is designed to work with real codebases by iteratively:

```text
Generate → Execute → Analyze → Retrieve → Repair → Verify
```

When a program fails, ODIN analyzes the error and, when necessary, retrieves relevant programming knowledge such as documentation, API references, and code examples to help generate a better repair.

The project also investigates **whether and when RAG improves LLM-based self-debugging** compared to relying only on execution feedback.

## Tech Stack

* Python
# ODIN

ODIN is an agentic self-debugging demo: Gemini generates Python code, a subprocess executes it, and LangGraph routes failed runs back through repair.

## Setup

Requires Python 3.12+ and `uv`.

```powershell
uv venv
uv pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GEMINI_API_KEY` in `.env`.

## Run

```powershell
uv run python scripts/demo.py "Write Python code that divides 10 by zero, then repair it"
uv run streamlit run frontend/app.py
```

The architecture is a small Generate -> Execute -> Analyze loop: LangGraph owns state and routing, Gemini generates or repairs code, the executor runs it with a timeout, and the classifier labels failures.

RAG, FastAPI, and experiments are planned for later phases.
