import logging.handlers
import os
from datetime import datetime, UTC
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PlaywrightURLLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, Message, User
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters, \
    CallbackQueryHandler, CallbackContext

from infobase.lileg_agent import cached_embedder

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


async def reply_message(user: User, message: Message):
    keyboard_markup = InlineKeyboardMarkup([
        [InlineKeyboardButton("Delete", callback_data=f"delete:{message.message_id}"), ],
    ])

    await message.reply_text(
        f'Message ({message.message_id}) was successfully ingested to the user collection ({user.id})',
        reply_to_message_id=message.message_id,
        reply_markup=keyboard_markup,
    )


def split_documents(documents: list[Document]) -> list[Document]:
    # Telegram has a maximum message length of 4096 characters.
    # The GPT-3.5-turbo context window is 4096 tokens.
    # The token is around 4 characters.
    # We want to allow 4096 characters for a user prompt which is 1024 tokens.
    # Additionally, we can provide 3072 tokens as prompt snippets for llm context.
    # For example 12 snippets will result in 256 tokens or 1024 characters per snippet.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)


def safe_detect_language(text: str):
    language = "unknown"
    try:
        language = detect(text)
    except Exception as ex:
        # langdetect.lang_detect_exception.LangDetectException: No features in text.
        logger.warning("Failed to detect language! %s", ex)

    return language


async def ingest_internal(user: User, message: Message, documents: list[Document]):
    assert all(info.page_content != "" for info in documents)

    message_id_as_str = str(message.message_id)

    logger.info("Splitting documents %s", message_id_as_str)
    documents = split_documents(documents)

    current_datetime_as_str = str(datetime.now(UTC))
    for document in documents:
        document.metadata["message_id"] = message_id_as_str
        document.metadata["message_date"] = str(message.date)

        document.metadata["language"] = safe_detect_language(document.page_content)
        document.metadata["ingested_on"] = current_datetime_as_str

    vector_store = Chroma(
        collection_name=str(user.id),
        embedding_function=cached_embedder,
        persist_directory=".chroma",
    )

    logger.info("Removing existing documents %s", message_id_as_str)
    vector_store.delete(where={"message_id": message_id_as_str})

    logger.info("Writing documents %s", message_id_as_str)
    await vector_store.aadd_documents(documents)


async def ingest_text(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user

    logger.info("Start indexing text from message id %s", message.message_id)

    documents = [
        Document(page_content=message.text, metadata={"source": str(message.message_id)})
    ]
    [x.metadata.update({"source_type": "message"}) for x in documents]
    await ingest_internal(user, message, documents)

    logger.info("Finish indexing text from message id %s", message.message_id)

    await reply_message(user, message)


async def ingest_url(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user

    logger.info("Start indexing url from message id %s", message.message_id)

    documents = await PlaywrightURLLoader(
        urls=message.text.splitlines(), remove_selectors=["header", "footer"]
    ).aload()

    [x.metadata.update({"source_type": "url"}) for x in documents]
    await ingest_internal(update.effective_user, message, documents)

    logger.info("Finish indexing url from message id %s", message.message_id)

    await reply_message(user, message)


async def ingest_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user

    logger.info("Start indexing file from message id %s", message.message_id)

    local_file_path = Path(".documents") / str(update.effective_user.id) / message.document.file_name
    local_file_path.parent.mkdir(parents=True, exist_ok=True)

    file = await context.bot.get_file(message.document)
    await file.download_to_drive(local_file_path)

    message = (update.message or update.edited_message)

    logger.info("Loading content from pdf %s", local_file_path)
    documents = await PyPDFLoader(local_file_path).aload()

    [x.metadata.update({"source_type": "file"}) for x in documents]
    await ingest_internal(update.effective_user, message, documents)

    logger.info("Finish indexing file from message id %s", message.message_id)

    await reply_message(user, message)


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


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
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
