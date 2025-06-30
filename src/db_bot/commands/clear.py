import logging
import os

from langchain_chroma import Chroma
from telegram import Update
from telegram.ext import ContextTypes

from src.common.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)

embedding_service = EmbeddingService()


async def clear(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("Removing existing messages from message id %s", message.message_id)

    vector_store = Chroma(
        collection_name=str(message.chat_id),
        embedding_function=embedding_service.embedding,
        persist_directory=os.getenv('CHROMA_PERSIST_DIR'),
    )
    vector_store.reset_collection()

    await update.effective_message.reply_text(
        f"Your data have been removed from the database!",
        reply_to_message_id=update.effective_message.message_id
    )
