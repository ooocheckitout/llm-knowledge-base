import logging
import os

from langchain_chroma import Chroma
from telegram import Document

from src.search_bot.constants.embeddings import cached_embeddings

logger = logging.getLogger(__name__)


async def similarity_search(collection_name: str, query: str, n_results: int) -> list[Document]:
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=cached_embeddings,
        persist_directory=os.getenv('CHROMA_PERSIST_DIR'),
    )
    return await vector_store.asimilarity_search(query=query, k=n_results)