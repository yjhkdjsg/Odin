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
* LangGraph
* LangChain
* Gemini
* ChromaDB
* Sentence Transformers
* FastAPI
* Git

## Status

Under Development.
