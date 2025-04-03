import logging.handlers
import os

import requests
import telegramify_markdown
from dotenv import load_dotenv
from langchain_chroma import Chroma
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters, CallbackContext, \
    CallbackQueryHandler

from infobase.lileg_agent import cached_embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()


async def welcome(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("welcome {} ({})".format(update.effective_user.full_name, update.effective_user.id))
    await update.message.reply_text(
        "\n".join([
            f"Welcome, {update.effective_user.first_name}!",
            "Send a message to search the the database.",
        ]),
        reply_to_message_id=update.message.message_id
    )


async def similarity_search(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user

    logger.info("Searching for message id %s", message.message_id)

    vector_store = Chroma(
        collection_name=str(user.id),
        embedding_function=cached_embedder,
        persist_directory=".chroma",
    )
    documents = vector_store.similarity_search(query=message.text, k=12)

    logger.info("Replying for message id %s", message.message_id)

    if not any(documents):
        await message.reply_text(f'No results found 😔', reply_to_message_id=message.message_id)

    for document in documents:
        await message.reply_text(document.page_content, reply_to_message_id=message.message_id)


async def search_llm(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("Searching for message id %s", message.message_id)

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    logger.info("Prompting for message id %s", message.message_id)
    response = requests.post(
        f"http://localhost:8000/users/{user_id}/chats/{chat_id}/complete",
        json={'question': message.text}
    )

    if not response.ok:
        raise Exception(
            "Failed to search from message id %s! [%s]: %s",
            update.effective_message.message_id, response.status_code, response.text
        )

    completion = response.json()

    logger.info("Replying for message id %s", message.message_id)
    markdown_content = telegramify_markdown.markdownify(completion["answer"])

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


async def keyboard_callback(update: Update, _: CallbackContext) -> None:
    query = update.callback_query

    logger.info("Received Keyboard callback %s", query.data)

    command, message_id = query.data.split(":")
    if command == "good" or command == "bad":
        logger.info("User review for message %s", message_id)

        await query.answer(f"User review '{command}' was saved!")
        await query.edit_message_reply_markup(reply_markup=None)
    else:
        await query.answer(f"Not supported!")


def error_handler(update: Update, context: CallbackContext) -> None:
    logger.warning(f'Update "{update}" caused error "{context.error}"')


TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_SEARCH_BOT_TOKEN')
app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

app.add_handler(CommandHandler("start", welcome))
app.add_handler(CommandHandler("similarity", similarity_search))
app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), search_llm))
app.add_handler(CallbackQueryHandler(keyboard_callback))
app.add_error_handler(error_handler)

app.run_polling()
