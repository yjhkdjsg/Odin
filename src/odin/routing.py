from odin.state import GraphState


def route_after_execute(state: GraphState) -> str:
    if state["success"] or state["iteration"] >= state["max_iterations"]:
        return "end"
    return "generate"