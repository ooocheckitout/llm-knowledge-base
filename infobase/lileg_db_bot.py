import logging.handlers
import os
from datetime import datetime, UTC
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PlaywrightURLLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, Message, User
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters, \
    CallbackQueryHandler, CallbackContext

from infobase.shared import CHROMA_CLIENT, EMBEDDINGS

logger = logging.getLogger(__name__)


def preview(text: str):
    return text[0:100].replace("\n", " ")


async def welcome(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("welcome {} ({})".format(update.effective_user.full_name, update.effective_user.id))
    await update.message.reply_text(
        "\n".join([
            f"Welcome, {update.effective_user.first_name}!",
            "Send a message that you want to be indexed in the database."
        ]),
        reply_to_message_id=update.message.message_id
    )


async def echo(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("echo {} ({})".format(update.effective_user.full_name, update.effective_user.id))
    await update.message.reply_text(
        f"Received a message {update.message.message_id}!\n\n{update.message.text}",
        reply_to_message_id=update.message.message_id
    )


async def clear(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    collection_name = str(update.effective_user.id)

    if collection_name in CHROMA_CLIENT.list_collections():
        logger.info("Removing existing messages from collection %s", collection_name)
        CHROMA_CLIENT.delete_collection(collection_name)

    await update.message.reply_text(
        f"Your data have been removed from the database!",
        reply_to_message_id=update.message.message_id
    )


def ingest_internal(effective_user_id: str, message_id: str, message_text: str, metadata: dict[str, str]):
    vector_store = Chroma(
        collection_name=effective_user_id,
        embedding_function=EMBEDDINGS,
        client=CHROMA_CLIENT
    )
    default_metadata = {
        "message_id": message_id,
        "ingested_on": str(datetime.now(UTC)),
        "language": detect(message_text),
    }

    logger.info("Splitting message %s into chunks", message_id)

    # Telegram has a maximum message length of 4096 characters. As well as GPT-3.5-turbo context window.
    # We need to allow at around 1024 characters for a user prompt.
    # Additionally, we want to provide around 4 snippets to enhance the llm prompt.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=768,
        chunk_overlap=100
    )

    texts = text_splitter.create_documents([message_text], [default_metadata | metadata])

    logger.info(
        "Ingesting message %s into collection %s with metadata %s",
        message_id, effective_user_id, metadata
    )
    vector_store.delete(where={"$or": [{"message_id": message_id}, {"url": message_text}]})
    vector_store.add_documents(documents=texts)


async def reply_message(message: Message, user: User):
    keyboard_markup = InlineKeyboardMarkup([
        [InlineKeyboardButton("Delete", callback_data=f"delete:{message.message_id}"), ],
    ])

    await message.reply_text(
        f'Message ({message.message_id}) was successfully ingested to the user collection ({user.id})',
        reply_to_message_id=message.message_id,
        reply_markup=keyboard_markup,
    )


async def ingest_text(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    metadata = {
        "source": "telegram",
        "message_sent_on": str(message.date),
    }
    ingest_internal(str(update.effective_user.id), str(message.message_id), message.text, metadata)
    await reply_message(message, update.effective_user)


async def ingest_url(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("Loading content from url %s", message.text)
    documents = await PlaywrightURLLoader(urls=[message.text], remove_selectors=["header", "footer"]).aload()

    for doc in documents:
        metadata = doc.metadata | {
            "source": "telegram",
            "message_sent_on": str(message.date),
        }
        ingest_internal(str(update.effective_user.id), str(message.message_id), doc.page_content, metadata)

    await reply_message(message, update.effective_user)


async def ingest_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    file = await context.bot.get_file(update.message.document)

    file_path = Path(".documents") / str(update.effective_user.id) / update.message.document.file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    await file.download_to_drive(file_path)

    message = (update.message or update.edited_message)

    logger.info("Loading content from pdf %s", message.text)
    documents = await PyPDFLoader(file_path).aload()

    for doc in documents:
        metadata = doc.metadata | {
            "source": "telegram",
            "message_sent_on": str(message.date),
        }
        ingest_internal(str(update.effective_user.id), str(message.message_id), doc.page_content, metadata)

    await reply_message(message, update.effective_user)


async def keyboard_callback(update: Update, _: CallbackContext) -> None:
    query = update.callback_query

    logger.info("Received Keyboard callback %s", query.data)

    collection_name = str(update.effective_user.id)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=EMBEDDINGS,
        client=CHROMA_CLIENT
    )

    command, message_id = query.data.split(":")
    if command == "delete":
        logger.info("Deleting message %s from collection %s", message_id, collection_name)
        vector_store.delete(where={"message_id": message_id})
        await query.answer(f"Message {message_id} deleted!")
        await query.delete_message()
    else:
        await query.answer(f"Not supported!")


TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_DB_BOT_TOKEN')
app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

app.add_handler(CommandHandler("start", welcome))
app.add_handler(CommandHandler("clear", clear))
app.add_handler(
    MessageHandler(filters.TEXT & (~filters.COMMAND) & (~filters.REPLY) & filters.Entity("url"), ingest_url)
)
app.add_handler(
    MessageHandler(filters.TEXT & (~filters.COMMAND) & (~filters.REPLY) & (~filters.Entity("url")), ingest_text)
)
app.add_handler(MessageHandler(filters.Document.ALL, ingest_file))
app.add_handler(CallbackQueryHandler(keyboard_callback))

app.run_polling()
