from typing import Literal
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage

from .state import HRState
from .config import llm, SALARY_DOMAINS
from .retriever import retriever, tavily
from .logger import get_logger

log = get_logger("nodes")

def _format_docs(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        entry = "\n".join([
            f"=== ВАКАНСИЯ {i} ===",
            f"ID: {m.get('id', 'N/A')}",
            f"Должность: {m.get('professional_roles', 'N/A')}",
            f"Компания: {m.get('employer', 'N/A')}",
            f"Локация: {m.get('area', 'N/A')}",
            f"Скиллы: {', '.join(m.get('extracted_skills', []))}",
            f"Описание: {doc.page_content[:300]}...",
        ])
        parts.append(entry)
    return "\n\n".join(parts)


def _parse_tavily(raw, max_chars: int = 1500) -> str:
    if isinstance(raw, str):
        return raw[:max_chars]
    results = raw.get("results", [])
    parts = [f"Source: {r['title']}\n{r['content']}" for r in results]
    return "\n\n".join(parts)[:max_chars]


class QueryClassification(BaseModel):
    query_type: Literal["vacancies", "salary", "both", "smalltalk"]


_classifier_llm = llm.with_structured_output(QueryClassification)


def classify_node(state: HRState) -> dict:
    query = state["messages"][-1].content
    log.info("classify_node ← '%s'", query[:80])

    result = _classifier_llm.invoke([
        SystemMessage(content=(
            "Определи тип запроса пользователя:\n"
            "- vacancies  — ищет вакансии или работу\n"
            "- salary     — спрашивает зарплаты или рынок труда\n"
            "- both       — нужны и вакансии, и зарплаты/рынок\n"
            "- smalltalk  — не про работу совсем\n"
            "Только схема, без объяснений."
        )),
        HumanMessage(content=query),
    ])
    log.info("classify_node → query_type='%s'", result.query_type)
    return {"query_type": result.query_type}


def qdrant_node(state: HRState) -> dict:
    query = state["messages"][-1].content
    log.info("qdrant_node ← '%s'", query[:80])

    docs = retriever.invoke(query)
    results = _format_docs(docs)
    log.info("qdrant_node → найдено %d вакансий (%d символов)", len(docs), len(results))
    log.debug("qdrant_node  doc ids: %s", [d.metadata.get("id") for d in docs])
    return {"qdrant_results": results}


def market_node(state: HRState) -> dict:
    query = state["messages"][-1].content
    log.info("market_node ← '%s'", query[:80])

    raw = tavily.invoke({"query": query, "include_domains": SALARY_DOMAINS, "search_depth": "basic"})
    results = _parse_tavily(raw)
    log.info("market_node → %d символов", len(results))
    return {"market_results": results}


def answer_node(state: HRState) -> dict:
    has_qdrant = bool(state.get("qdrant_results"))
    has_market = bool(state.get("market_results"))
    log.info(
        "answer_node ← qdrant=%s  market=%s  history_len=%d",
        has_qdrant, has_market, len(state["messages"]),
    )

    context_parts = []
    if has_qdrant:
        context_parts.append(f"ВАКАНСИИ ИЗ БАЗЫ:\n{state['qdrant_results']}")
    if has_market:
        context_parts.append(f"ДАННЫЕ РЫНКА ТРУДА:\n{state['market_results']}")

    context = "\n\n".join(context_parts)
    system_content = "Ты HR-ассистент."
    if context:
        system_content += f"\n\nОтвечай строго на основе контекста:\n\n{context}"

    response = llm.invoke([SystemMessage(content=system_content)] + list(state["messages"]))
    log.info("answer_node → ответ %d символов", len(response.content))
    return {"messages": [response]}
