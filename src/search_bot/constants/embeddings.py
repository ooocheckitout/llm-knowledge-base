import logging
import os

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

logger.info(f"Initializing OllamaEmbeddings({os.getenv('OLLAMA_EMBEDDING_MODEL')})")
embeddings = OllamaEmbeddings(
    model=os.getenv('OLLAMA_EMBEDDING_MODEL'),
    base_url=os.getenv('OLLAMA_BASE_URL'),
)

logger.info(f"Initializing CacheBackedEmbeddings({os.getenv('EMBEDDINGS_CACHE_DIR')})")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings, LocalFileStore(os.getenv('EMBEDDINGS_CACHE_DIR')),
    namespace=embeddings.model
)
