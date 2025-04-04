import logging.handlers
import os

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# LOGGING
file_handler = logging.handlers.RotatingFileHandler(f".logs/{os.path.basename(__file__)}.log", backupCount=10)
file_handler.doRollover()

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, logging.StreamHandler()]
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ENVIRONMENT VARIABLES
load_dotenv()

# DATABASE AND EMBEDDINGS
CHROMA_CLIENT = chromadb.PersistentClient(".chroma")

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

collection_name = str(213260575)
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=EMBEDDINGS,
    client=CHROMA_CLIENT
)


def preview(text: str):
    return text[0:100].replace("\n", " ")


retriever = vector_store.as_retriever(search_kwargs={'filter': {"source": "telegram"}})

query = "chunk"
results = vector_store.similarity_search_with_score(query, filter={"source": "telegram"})
for doc, score in results:
    logger.info(f"* [SIM={score:3f}; LENGTH={len(doc.page_content)}] [{doc.metadata}]")


