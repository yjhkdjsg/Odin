from langgraph.graph import END, START, StateGraph

from odin.nodes import analyze_node, execute_node, generate_node
from odin.routing import route_after_execute
from odin.state import GraphState


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("generate", generate_node)
    graph.add_node("execute", execute_node)
    graph.add_node("analyze", analyze_node)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "execute")
    graph.add_edge("execute", "analyze")
    graph.add_conditional_edges("analyze", route_after_execute, {"generate": "generate", "end": END})
    return graph.compile()