import logging.handlers
import os
from typing import Optional

from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


class ChatOpenRouter(ChatOpenAI):
    def __init__(self,
                 model: str,
                 api_key: Optional[str] = None,
                 base_url: str = "https://openrouter.ai/api/v1",
                 **kwargs):
        api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        super().__init__(base_url=base_url, api_key=api_key, model=model, **kwargs)


def preview(text: str):
    return text[0:100].replace("\n", " ")


def configure_logging(name: str):
    local_logs_path = Path(".logs") / f"{name}.log"
    local_logs_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(local_logs_path, backupCount=10)
    file_handler.doRollover()

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, logging.StreamHandler()]
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)


load_dotenv()

CHROMA_CLIENT_DIR = ".chroma"

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

LLM = ChatOpenRouter(
    model="deepseek/deepseek-chat-v3-0324:free",
    temperature=0.7
)

DEBUG_USER_ID = "213260575"
