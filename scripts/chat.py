"""Диалоговый режим: While True + история сообщений между ходами."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_core.messages import HumanMessage, AIMessage
from src.graph import app
from src.logger import get_logger

log = get_logger("chat")

SEPARATOR = "─" * 60
EXIT_COMMANDS = {"exit", "quit", "выход", "q", ":q"}


def main() -> None:
    log.info("диалоговый режим запущен")
    print(SEPARATOR)
    print("  HR-ассистент  |  введите 'exit' для выхода")
    print(SEPARATOR)

    history: list = []
    turn = 0

    while True:
        try:
            user_input = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[прерывание]")
            break

        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            log.info("сессия завершена пользователем  (ходов: %d)", turn)
            print("До свидания!")
            break

        turn += 1
        log.info("--- ход %d ---", turn)

        history.append(HumanMessage(content=user_input))

        result = app.invoke(
            {
                "messages":       history,
                "query_type":     "",
                "qdrant_results": "",
                "market_results": "",
            },
            config={"run_name": f"hr_chat_turn_{turn}"},
        )

        # обновляем историю — граф добавил AI-ответ через add_messages
        history = list(result["messages"])
        answer = history[-1].content

        log.info("ход %d завершён  (query_type='%s')", turn, result["query_type"])
        print(f"\nАссистент: {answer}")
        print(SEPARATOR)


if __name__ == "__main__":
    main()
