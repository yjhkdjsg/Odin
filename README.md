# ODIN

### Orchestrated Debugging and Intelligent Navigation

ODIN is an agentic self-debugging framework for automated Python code repair. For this first checkpoint, Gemini generates code, ODIN executes it, classifies failures, and sends failed code back to Gemini for repair.

## Current Workflow

```text
Generate -> Execute -> Analyze -> Repair
					^          |
					|----------|
```

The workflow stops when the generated program succeeds or the maximum number of iterations is reached.

## Tech Stack

- Python 3.12+
- uv for environment and package management
- LangGraph for the agent workflow and state transitions
- Gemini through the `google-genai` SDK for code generation and repair
- Python subprocess execution with a five-second timeout
- Streamlit for the live demonstration UI

## Project Structure

```text
src/odin/
├── state.py             # LangGraph state definition
├── executor.py          # Subprocess code execution
├── error_classifier.py  # Traceback categorization
├── llm.py               # Gemini generation and repair
├── nodes.py             # Generate, execute, and analyze nodes
├── routing.py           # Deterministic loop routing
└── graph.py             # Compiled LangGraph workflow

frontend/app.py          # Streamlit interface
scripts/demo.py          # Command-line entry point
```

## Setup

Requires Python 3.12+ and `uv`.

```powershell
uv venv
uv pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your Gemini API key to `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

Never commit `.env` or expose its API key. `.env.example` is safe to commit because it contains only the variable name and an empty value.

## Run the CLI Demo

Pass a task description in quotes:

```powershell
uv run python scripts/demo.py "Write a Python program that calculates an average and repair any runtime errors"
```

The CLI prints the final success state, iteration count, error category, generated code, and any final traceback.

## Run the Streamlit App

```powershell
uv run streamlit run frontend/app.py
```

Open the local URL shown by Streamlit, enter a task, and select **Run**. The page displays the current graph stage, generated code for each iteration, error categories, and the final result.

## Error Categories

The current classifier maps common traceback types to:

- `syntax`
- `runtime`
- `import`
- `logical`
- `unknown`

## Known Limitation

Generated code runs in a subprocess with a timeout but is not sandboxed. This is an intentional limitation for the first checkpoint and is not suitable for untrusted code.

## Future Phases

RAG, document and code retrieval, ChromaDB, embeddings, FastAPI, broader evaluation experiments, and other production hardening are planned for later phases. They are intentionally outside the scope of this checkpoint.
