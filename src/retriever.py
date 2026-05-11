import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_tavily import TavilySearch
from qdrant_client import QdrantClient

from .config import COLLECTION_NAME, DB_PATH, SALARY_DOMAINS
from .logger import get_logger

log = get_logger("retriever")

log.debug("подключение к Qdrant: %s", DB_PATH)
client = QdrantClient(path=DB_PATH)
log.info("Qdrant client готов  (collection=%s)", COLLECTION_NAME)

device = "mps" if torch.backends.mps.is_available() else "cpu"
log.debug("загрузка embedding-модели на %s", device)
embeddings_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)
log.info("embeddings готовы  (intfloat/multilingual-e5-small, device=%s)", device)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings_model,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
log.info("retriever готов  (k=3)")

tavily = TavilySearch(max_results=3, topic="general")
log.info("Tavily готов")
