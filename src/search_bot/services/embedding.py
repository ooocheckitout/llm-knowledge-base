import logging
import os

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self._embedding = None
        self.providers = {
            "ollama": self.initialize_ollama,
            "openai": self.initialize_openai,
            "huggingface": self.initialize_huggingface,
        }

    @property
    def embedding(self):
        if not self._embedding:
            self._embedding = self.cached(self.providers[os.getenv('EMBEDDING_PROVIDER')]())

        return self._embedding

    @staticmethod
    def cached(embeddings: Embeddings):
        logger.info(f"Initializing CacheBackedEmbeddings({os.getenv('EMBEDDINGS_CACHE_DIR')})")
        return CacheBackedEmbeddings.from_bytes_store(
            embeddings, LocalFileStore(os.getenv('EMBEDDINGS_CACHE_DIR')), namespace=getattr(embeddings, 'model', embeddings.model_name)
        )

    @staticmethod
    def initialize_ollama():
        logger.info(f"Initializing OllamaEmbeddings({os.getenv('EMBEDDING_MODEL')})")
        return OllamaEmbeddings(
            model=os.getenv('EMBEDDING_MODEL'),
            base_url=os.getenv('OLLAMA_BASE_URL')
        )

    @staticmethod
    def initialize_openai():
        logger.info(f"Initializing OpenAIEmbeddings({os.getenv('EMBEDDING_MODEL')})")
        return OpenAIEmbeddings(
            model=os.getenv('EMBEDDING_MODEL'),
            base_url=os.getenv('OPENAI_BASE_URL'),
            api_key=os.getenv('OPENAI_API_KEY')
        )

    @staticmethod
    def initialize_huggingface():
        logger.info(f"Initializing HuggingFaceEmbeddings({os.getenv('EMBEDDING_MODEL')})")
        return HuggingFaceEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
