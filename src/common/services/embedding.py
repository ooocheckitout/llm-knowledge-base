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
        providers = {
            "ollama": self.initialize_ollama,
            "openai": self.initialize_openai,
            "huggingface": self.initialize_huggingface,
        }
        model_name = os.getenv('EMBEDDING_MODEL', 'all-minilm:l6-v2')
        cache_location = os.getenv('EMBEDDINGS_CACHE_DIR', '.cache/embeddings')
        self._embedding = self.cached(
            providers[os.getenv('EMBEDDING_PROVIDER', 'ollama')](model_name),
            cache_location
        )

    @property
    def embedding(self):
        return self._embedding

    def vectorize(self, texts: list[str]):
        return self.embedding.embed_documents(texts)

    @staticmethod
    def cached(embeddings: Embeddings, location: str):
        logger.info(f"Initializing CacheBackedEmbeddings({location})")
        return CacheBackedEmbeddings.from_bytes_store(
            embeddings, LocalFileStore(location)
        )

    @staticmethod
    def initialize_ollama(model_name: str):
        logger.info(f"Initializing OllamaEmbeddings({model_name})")
        return OllamaEmbeddings(
            model=model_name,
            base_url=os.getenv('OLLAMA_BASE_URL')
        )

    @staticmethod
    def initialize_openai(model_name: str):
        logger.info(f"Initializing OpenAIEmbeddings({model_name})")
        return OpenAIEmbeddings(
            model=model_name,
            base_url=os.getenv('OPENAI_BASE_URL'),
            api_key=os.getenv('OPENAI_API_KEY')
        )

    @staticmethod
    def initialize_huggingface(model_name: str):
        logger.info(f"Initializing HuggingFaceEmbeddings({model_name})")
        return HuggingFaceEmbeddings(
            model=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
