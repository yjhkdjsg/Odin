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
