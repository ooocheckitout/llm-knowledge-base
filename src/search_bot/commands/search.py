import logging
import os
from typing import TypedDict

import telegramify_markdown
from src.search_bot.constants.embeddings import cached_embeddings
from src.search_bot.constants.chat import llm
from src.search_bot.constants.template import prompt
from src.search_bot.constants.history import get_message_history_by_session_id

from langchain_chroma import Chroma
from telegram import Update, Document, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


async def similarity_search(collection_name: str, query: str, n_results: int) -> list[Document]:
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=cached_embeddings,
        persist_directory=os.getenv('CHROMA_PERSIST_DIR'),
    )
    return await vector_store.asimilarity_search(query=query, k=n_results)


async def search(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("Searching in %s for message id %s", message.chat_id, message.message_id)
    documents = await similarity_search(str(message.chat_id), message.text, 3)

    logger.info("Replying for message id %s", message.message_id)
    if not any(documents):
        await message.reply_text(f'No results found 😔', reply_to_message_id=message.message_id)

    for document in documents:
        await message.reply_text(document.page_content, reply_to_message_id=message.message_id)


async def search_llm(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("Searching in %s for message id %s", message.chat_id, message.message_id)

    documents = await similarity_search(str(message.chat_id), message.text, 12)

    logger.info("Enriching context with %s documents for message id %s", len(documents), message.message_id)

    context_template = """
The source for the following context is {source_type} {source}:
"{content}" 
        """

    context = "\n".join([
        context_template
        .replace("{content}", x.page_content)
        .replace("{source_type}", x.metadata["source_type"])
        .replace("{source}", x.metadata["source"])
        for x in documents
    ])
    missing_context = r"No context is available. Try adding more information to @lileg_db_bot."

    logger.info("Retrieving history for message id %s", message.message_id)
    session_history = get_message_history_by_session_id(str(message.chat_id))

    logger.info(
        "Enriching history with %s messages for message id %s", len(session_history.messages), message.message_id
    )
    history = "\n".join([f"{x.type}: \"{x.content}\"" for x in session_history.messages])

    logger.info("Prompting for message id %s", message.message_id)
    chain = prompt | llm
    completion = chain.invoke({"question": message.text, "history": history, "context": context or missing_context})

    logger.info("Saving history for message id %s", message.message_id)
    session_history.add_user_message(message.text)
    session_history.add_ai_message(completion.content)

    logger.info("Replying for message id %s", message.message_id)
    markdown_content = telegramify_markdown.markdownify(completion.content)

    keyboard_markup = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Good response", callback_data=f"good:{message.message_id}"),
            InlineKeyboardButton("Bad response", callback_data=f"bad:{message.message_id}"),
        ],
    ])

    await message.reply_markdown_v2(
        markdown_content,
        reply_to_message_id=message.message_id,
        reply_markup=keyboard_markup,
    )
