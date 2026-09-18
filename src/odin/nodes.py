from odin.error_classifier import classify_error
from odin.executor import run_code
from odin.llm import generate_code
from odin.state import GraphState


def generate_node(state: GraphState) -> dict:
    code = generate_code(state["task"], state["code"], state["error"])
    return {"code": code, "iteration": state["iteration"] + 1, "history": [{"stage": "generate", "iteration": state["iteration"] + 1, "code": code}]}


def execute_node(state: GraphState) -> dict:
    result = run_code(state["code"])
    error = result.stderr if not result.success else ""
    return {"success": result.success, "error": error, "history": [{"stage": "execute", "success": result.success, "stdout": result.stdout, "stderr": result.stderr}]}


def analyze_node(state: GraphState) -> dict:
    category = classify_error(state["error"]) if state["error"] else ""
    return {"error_category": category, "history": [{"stage": "analyze", "error_category": category}]}