from langgraph.graph import StateGraph, START, END

from .state import HRState
from .nodes import classify_node, qdrant_node, market_node, answer_node
from .edges import route_after_classify, route_after_qdrant
from .logger import get_logger

log = get_logger("graph")


def build_graph():
    log.debug("сборка графа…")
    graph = StateGraph(HRState)

    graph.add_node("classify", classify_node)
    graph.add_node("qdrant",   qdrant_node)
    graph.add_node("market",   market_node)
    graph.add_node("answer",   answer_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify)
    graph.add_conditional_edges("qdrant",   route_after_qdrant)
    graph.add_edge("market", "answer")
    graph.add_edge("answer", END)

    app = graph.compile()
    log.info("граф скомпилирован  (узлы: classify → qdrant/market/answer)")
    return app


app = build_graph()
