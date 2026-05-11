from .state import HRState
from .logger import get_logger

log = get_logger("edges")


def route_after_classify(state: HRState) -> str:
    qt = state["query_type"]
    if qt == "salary":
        dest = "market"
    elif qt == "smalltalk":
        dest = "answer"
    else:
        dest = "qdrant"
    log.info("route_after_classify: '%s' → '%s'", qt, dest)
    return dest


def route_after_qdrant(state: HRState) -> str:
    dest = "market" if state["query_type"] == "both" else "answer"
    log.info("route_after_qdrant: query_type='%s' → '%s'", state["query_type"], dest)
    return dest
