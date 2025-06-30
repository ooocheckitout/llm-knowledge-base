import logging
import os

from langchain_chroma import Chroma
from telegram import Update
from telegram.ext import CallbackContext

from src.common.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)

embedding_service = EmbeddingService()

async def keyboard_callback(update: Update, _: CallbackContext) -> None:
    query = update.callback_query

    logger.info("Received Keyboard callback %s", query.data)

    command, message_id = query.data.split(":")
    if command == "delete":
        logger.info("Deleting message %s", message_id)

        vector_store = Chroma(
            collection_name=str(query.message.chat.id),
            embedding_function=embedding_service.embedding,
            persist_directory=os.getenv('CHROMA_PERSIST_DIR'),
        )
        vector_store.delete(where={"message_id": message_id})

        await query.answer(f"Message {message_id} deleted!")
        await query.delete_message()
    else:
        await query.answer(f"Not supported!")