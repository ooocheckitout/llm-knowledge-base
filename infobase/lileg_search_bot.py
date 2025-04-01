import logging.handlers
import os

import requests
import telegramify_markdown
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters, CallbackContext

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

    logger.info("Searching for message id %s", message.message_id)

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    response = requests.post(
        f'http://localhost:8000/users/{user_id}/chats/{chat_id}/similarity',
        json={'query': message.text, 'n_results': 12}
    )

    if not response.ok:
        raise Exception(
            "Failed to search from message id %s! [%s]: %s",
            update.effective_message.message_id, response.status_code, response.text
        )

    logger.info("Replying for message id %s", message.message_id)

    embeddings = response.json()

    if not any(embeddings):
        await message.reply_text(f'No results found 😔', reply_to_message_id=message.message_id)

    for embedding in embeddings:
        await message.reply_text(embedding["content"], reply_to_message_id=message.message_id)


async def search_llm(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("Searching for message id %s", message.message_id)

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    context = r"If the user doesn't have any context he should try messaging to @lileg\_db\_bot."

    logger.info("Prompting for message id %s", message.message_id)
    response = requests.post(
        f"http://localhost:8000/users/{user_id}/chats/{chat_id}/complete",
        json={'question': context + " " + message.text}
    )

    if not response.ok:
        raise Exception(
            "Failed to search from message id %s! [%s]: %s",
            update.effective_message.message_id, response.status_code, response.text
        )

    completion = response.json()

    logger.info("Replying for message id %s", message.message_id)
    markdown_content = telegramify_markdown.markdownify(completion["answer"])
    await message.reply_markdown_v2(markdown_content, reply_to_message_id=message.message_id)


def error_handler(update: Update, context: CallbackContext) -> None:
    logger.warning(f'Update "{update}" caused error "{context.error}"')


TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_SEARCH_BOT_TOKEN')
app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

app.add_handler(CommandHandler("start", welcome))
app.add_handler(CommandHandler("similarity", similarity_search))
app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), search_llm))
app.add_error_handler(error_handler)

app.run_polling()
