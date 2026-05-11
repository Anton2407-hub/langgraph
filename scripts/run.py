"""Разовый запуск графа через .invoke — точка входа для отладки."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_core.messages import HumanMessage
from src.graph import app
from src.logger import get_logger

log = get_logger("run")

QUERY = "Найди вакансии Data Scientist в спортивной компании"


def main(query: str = QUERY) -> None:
    log.info("запуск графа с запросом: '%s'", query)

    result = app.invoke(
        {
            "messages":       [HumanMessage(content=query)],
            "query_type":     "",
            "qdrant_results": "",
            "market_results": "",
        },
        config={"run_name": "hr_run"},
    )

    answer = result["messages"][-1].content
    log.info("готово  (query_type='%s')", result["query_type"])
    print("\n" + "=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else QUERY
    main(query)
