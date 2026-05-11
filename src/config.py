import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq, AuthenticationError
from langchain_groq import ChatGroq

from .logger import get_logger

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

log = get_logger("config")

COLLECTION_NAME = "vacancies_e5_small_v1"
DB_PATH = "./data/db/vacancies_db"

SALARY_DOMAINS = [
    "levels.fyi",
    "glassdoor.com",
    "career.habr.com",
    "getmatch.ru",
    "talent.com",
]


def _pick_groq_key() -> str:
    candidates = [
        os.getenv("GROQ_API_KEY", ""),
        os.getenv("GROQ_API_KEY_2", ""),
        os.getenv("GROQ_API_KEY_3", ""),
        os.getenv("GROQ_API_KEY_4", ""),
    ]
    for key in candidates:
        key = key.strip()
        if not key:
            continue
        try:
            Groq(api_key=key).models.list()
            log.debug("Groq key OK: ...%s", key[-6:])
            return key
        except AuthenticationError:
            log.warning("Groq key невалиден: ...%s, пробую следующий", key[-6:])
    raise RuntimeError("Ни один GROQ_API_KEY не прошёл проверку — обновите .env")


log.debug("инициализация LLM: llama-3.3-70b-versatile")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=2000,
    api_key=_pick_groq_key(),
)
log.info("LLM готов")
