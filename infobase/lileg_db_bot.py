import logging.handlers
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.document_loaders import PlaywrightURLLoader, PyPDFLoader
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, Message, User
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters, \
    CallbackQueryHandler, CallbackContext

# configure_logging(os.path.basename(__file__))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()


async def welcome(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("welcome {} ({})".format(update.effective_user.full_name, update.effective_user.id))
    await update.effective_message.reply_text(
        "\n".join([
            f"Welcome, {update.effective_user.first_name}!",
            "Send a message that you want to be indexed in the database."
        ]),
        reply_to_message_id=update.effective_message.message_id
    )


async def echo(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("echo {} ({})".format(update.effective_user.full_name, update.effective_user.id))
    await update.effective_message.reply_text(
        f"Received a message {update.effective_message.message_id}!\n\n{update.effective_message.text}",
        reply_to_message_id=update.effective_message.message_id
    )


async def clear(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("Removing existing messages from message id %s", message.message_id)

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    response = requests.post(
        f'http://localhost:8000/users/{user_id}/chats/{chat_id}/forgetAll',
    )

    if not response.ok:
        raise Exception(
            "Failed to forget all from message id %s! [%s]: %s",
            message.message_id, response.status_code, response.text
        )

    await update.effective_message.reply_text(
        f"Your data have been removed from the database!",
        reply_to_message_id=update.effective_message.message_id
    )


async def reply_message(message: Message, user: User):
    keyboard_markup = InlineKeyboardMarkup([
        [InlineKeyboardButton("Delete", callback_data=f"delete:{message.message_id}"), ],
    ])

    await message.reply_text(
        f'Message ({message.message_id}) was successfully ingested to the user collection ({user.id})',
        reply_to_message_id=message.message_id,
        reply_markup=keyboard_markup,
    )


def enrich_metadata(documents: list[Document], message: Message, metadata: dict[str, str]) -> list[Document]:
    logger.info("Enhancing document metadata from message id %s", message.message_id)
    message_metadata = metadata | {
        "message_id": str(message.message_id),
        "message_date": str(message.date),
    }

    for document in documents:
        document.metadata.update(message_metadata)

        for k, v in document.metadata.items():
            if not isinstance(v, str):
                document.metadata[k] = str(v)

    return documents


async def ingest_text(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("Start indexing text from message id %s", message.message_id)

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    documents = [
        Document(page_content=message.text, metadata={"source": str(message.message_id)})
    ]
    documents = enrich_metadata(documents, message, {"source_type": "message"})
    response = requests.post(
        f'http://localhost:8000/users/{user_id}/chats/{chat_id}/remember',
        json=[{"content": x.page_content, "metadata": x.metadata} for x in documents]
    )

    if not response.ok:
        raise Exception(
            "Failed to remember from message id %s! [%s]: %s",
            message.message_id, response.status_code, response.text
        )

    logger.info("Finish indexing text from message id %s", message.message_id)

    await reply_message(message, update.effective_user)


async def ingest_url(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("Start indexing url from message id %s", message.message_id)

    documents = await PlaywrightURLLoader(
        urls=message.text.splitlines(), remove_selectors=["header", "footer"]
    ).aload()

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    documents = enrich_metadata(documents, message, {"source_type": "url"})
    response = requests.post(
        f'http://localhost:8000/users/{user_id}/chats/{chat_id}/remember',
        json=[{"content": x.page_content, "metadata": x.metadata} for x in documents]
    )

    if not response.ok:
        raise Exception(
            "Failed to remember from message id %s! [%s]: %s",
            message.message_id, response.status_code, response.text
        )

    logger.info("Finish indexing url from message id %s", message.message_id)

    await reply_message(message, update.effective_user)


async def ingest_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("Start indexing file from message id %s", message.message_id)

    local_file_path = Path(".documents") / str(update.effective_user.id) / message.document.file_name
    local_file_path.parent.mkdir(parents=True, exist_ok=True)

    file = await context.bot.get_file(message.document)
    await file.download_to_drive(local_file_path)

    message = (update.message or update.edited_message)

    logger.info("Loading content from pdf %s", local_file_path)
    documents = await PyPDFLoader(local_file_path).aload()

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    documents = enrich_metadata(documents, message, {"source_type": "file"})
    response = requests.post(
        f'http://localhost:8000/users/{user_id}/chats/{chat_id}/remember',
        json=[{"content": x.page_content, "metadata": x.metadata} for x in documents]
    )

    if not response.ok:
        raise Exception(
            "Failed to remember from message id %s! [%s]: %s",
            message.message_id, response.status_code, response.text
        )

    logger.info("Finish indexing file from message id %s", message.message_id)

    await reply_message(message, update.effective_user)


async def keyboard_callback(update: Update, _: CallbackContext) -> None:
    query = update.callback_query

    logger.info("Received Keyboard callback %s", query.data)

    command, message_id = query.data.split(":")
    if command == "delete":
        logger.info("Deleting message %s", message_id)

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        response = requests.post(
            f'http://localhost:8000/users/{user_id}/chats/{chat_id}/forget',
            json={"filter": {"message_id": message_id}}
        )

        if not response.ok:
            raise Exception(
                "Failed to forget all from message id %s! [%s]: %s",
                message_id, response.status_code, response.text
            )

        await query.answer(f"Message {message_id} deleted!")
        await query.delete_message()
    else:
        await query.answer(f"Not supported!")


def error_handler(update: Update, context: CallbackContext) -> None:
    logger.warning(f'Update "{update}" caused error "{context.error}"')


class EnsureSingleEntity(filters.Entity):
    def filter(self, message: Message) -> bool:
        return message.entities and all(entity.type == self.entity_type for entity in message.entities)


TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_DB_BOT_TOKEN')
app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

app.add_handler(CommandHandler("start", welcome))
app.add_handler(CommandHandler("clear", clear))
app.add_handler(MessageHandler(EnsureSingleEntity("url"), ingest_url))
app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), ingest_text))
app.add_handler(MessageHandler(filters.Document.ALL, ingest_file))
app.add_handler(CallbackQueryHandler(keyboard_callback))
app.add_error_handler(error_handler)

app.run_polling()
