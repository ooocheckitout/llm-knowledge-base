import logging.handlers
import os

import requests
import telegramify_markdown
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters, CallbackContext

from infobase.shared import preview, configure_logging

configure_logging(os.path.basename(__file__))

logger = logging.getLogger(__name__)


async def welcome(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("welcome {} ({})".format(update.effective_user.full_name, update.effective_user.id))
    await update.message.reply_text(
        "\n".join([
            f"Welcome, {update.effective_user.first_name}!",
            "Send a message to search the the database.",
        ]),
        reply_to_message_id=update.message.message_id
    )


async def similarity_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    collection_name = str(update.effective_user.id)
    logger.info("Performing similarity search for query %s", preview(message.text))
    similarity_response = requests.post('http://localhost:8000/similarity', data={
        'text': 1,
        'filter': {"user_id": collection_name},
        'n_results': 12
    }).json()

    if not similarity_response["results"]:
        await message.reply_text(f'No results found 😔', reply_to_message_id=message.message_id)

    for result in similarity_response["results"]:
        await message.reply_text(result, reply_to_message_id=message.message_id)


async def search_llm(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("search %s", preview(message.text))

    collection_name = str(update.effective_user.id)

    logger.info("Performing similarity search for query %s", preview(message.text))
    similarity_response = requests.post('http://localhost:8000/similarity', json={
        'text': message.text,
        'filter': {"user_id": collection_name},
        'n_results': 12
    })
    similarity_response = similarity_response.json()

    context = "\n\n".join(similarity_response["results"])
    if not context:
        context = (
            r"User have not added any context to the vector database. Tell him to add some context by messaging to @lileg\_db\_bot."
        )

    logger.info("Executing llm prompt for query %s", preview(message.text))
    llm_response = requests.post('http://localhost:8000/llm', json={
        'question': message.text, 'context': context, 'user_id': collection_name
    })
    llm_response = llm_response.json()

    logger.info("Sending a reply for query %s", preview(message.text))
    markdown_content = telegramify_markdown.markdownify(llm_response["answer"])
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
