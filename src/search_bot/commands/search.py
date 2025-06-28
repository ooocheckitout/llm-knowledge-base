import logging

from telegram import Update
from telegram.ext import ContextTypes

from src.search_bot.commands.similarity_search import similarity_search

logger = logging.getLogger(__name__)



async def search(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("Searching in %s for message id %s", message.chat_id, message.message_id)
    documents = await similarity_search(str(message.chat_id), message.text, 3)

    logger.info("Replying for message id %s", message.message_id)
    if not any(documents):
        await message.reply_text(f'No results found 😔', reply_to_message_id=message.message_id)

    for document in documents:
        await message.reply_text(document.page_content, reply_to_message_id=message.message_id)



