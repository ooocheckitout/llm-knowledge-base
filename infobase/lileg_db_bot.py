import hashlib
import logging.handlers
import os
from datetime import datetime, UTC
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PlaywrightURLLoader, PyPDFLoader
from langdetect import detect
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
    collection_name = str(update.effective_user.id)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=EMBEDDINGS,
        persist_directory=CHROMA_CLIENT_DIR,
    )

    logger.info("Removing existing messages from collection %s", collection_name)
    vector_store.delete_collection()

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


def hash_text(text: str, algorithm="sha256", encoding="utf-8") -> str:
    hasher = hashlib.new(algorithm)
    hasher.update(text.encode())
    return hasher.hexdigest()


def ingest_documents(collection_name: str, documents: list[Document], message: Message):
    if all(doc.page_content == "" for doc in documents):
        logger.info("Failed to fetch documents from message id %s", update.effective_message.message_id)
        raise Exception("All documents are empty")

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=EMBEDDINGS,
        persist_directory=CHROMA_CLIENT_DIR,
    )

    logger.info("Calculating hash from message id %s", message.message_id)
    message_text_hash = hash_text(message.text or message.document.file_name)

    language = "unknown"
    try:
        logger.info("Detecting language from message id %s", message.message_id)
        language = detect(message.text)
    except Exception:
        # langdetect.lang_detect_exception.LangDetectException: No features in text.
        logger.warning("Failed to detect language from message id %s", message.message_id)

    logger.info("Enhancing document metadata from message id %s", message.message_id)
    message_metadata = {
        "hash": message_text_hash,
        "message_id": str(message.message_id),
        "message_date": str(message.date),
        "ingested_on": str(datetime.now(UTC)),
        "language": language,
    }

    [x.metadata.update(message_metadata) for x in documents]


async def ingest_text(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("Start indexing text from message id %s", message.message_id)

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    response = requests.post(
        f'http://localhost:8000/users/{user_id}/chats/{chat_id}/vectorize',
        json=[{
            "content": message.text,
            "metadata": {"source": str(message.message_id)}
        }]
    )

    if not response.ok:
        raise Exception(
            "Failed to vectorize from message id %s! [%s]: %s",
            message.message_id, response.status_code, response.text
        )

    logger.info("Finish indexing text from message id %s", message.message_id)

    await reply_message(message, update.effective_user)


async def ingest_url(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    collection_name = str(update.effective_user.id)

    logger.info("Start indexing url from message id %s", update.effective_message.message_id)

    documents = await PlaywrightURLLoader(
        urls=update.effective_message.text.splitlines(), remove_selectors=["header", "footer"]
    ).aload()

    ingest_documents(collection_name, documents, message=update.effective_message)

    logger.info("Finish indexing url from message id %s", update.effective_message.message_id)

    await reply_message(update.effective_message, update.effective_user)


async def ingest_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    collection_name = str(update.effective_user.id)
    logger.info("Start indexing file from message id %s", update.effective_message.message_id)

    local_file_path = Path(".documents") / str(update.effective_user.id) / update.effective_message.document.file_name
    local_file_path.parent.mkdir(parents=True, exist_ok=True)

    file = await context.bot.get_file(update.effective_message.document)
    await file.download_to_drive(local_file_path)

    message = (update.message or update.edited_message)

    logger.info("Loading content from pdf %s", local_file_path)
    documents = await PyPDFLoader(local_file_path).aload()

    ingest_documents(collection_name, documents, message=update.effective_message)

    logger.info("Finish indexing file from message id %s", update.effective_message.message_id)

    await reply_message(message, update.effective_user)


async def keyboard_callback(update: Update, _: CallbackContext) -> None:
    query = update.callback_query

    logger.info("Received Keyboard callback %s", query.data)

    collection_name = str(update.effective_user.id)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=EMBEDDINGS,
        persist_directory=CHROMA_CLIENT_DIR,
    )

    command, message_id = query.data.split(":")
    if command == "delete":
        logger.info("Deleting message %s from collection %s", message_id, collection_name)
        vector_store.delete(where={"message_id": message_id})
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
