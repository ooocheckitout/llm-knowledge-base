import logging.handlers
import os
from datetime import datetime, UTC
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PlaywrightURLLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langdetect import detect
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

# LOGGING
file_handler = logging.handlers.RotatingFileHandler(f".logs/{os.path.basename(__file__)}.log", backupCount=10)
file_handler.doRollover()

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, logging.StreamHandler()]
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ENVIRONMENT VARIABLES
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_DB_BOT_TOKEN')

# DATABASE AND EMBEDDINGS
CHROMA_CLIENT = chromadb.PersistentClient(".chroma")

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)


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
    vector_store.delete(where={"message_id": message_id})
    vector_store.add_documents(documents=texts)


async def ingest_text(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = (update.message or update.edited_message)

    metadata = {
        "source": "telegram",
        "message_sent_on": str(message.date),
    }
    ingest_internal(str(update.effective_user.id), str(message.message_id), message.text, metadata)

    await message.reply_text(
        f'Message ({message.message_id}) was successfully ingested to the user collection ({update.effective_user.id})',
        reply_to_message_id=message.message_id,
    )


async def ingest_url(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = (update.message or update.edited_message)

    logger.info("Loading content from url %s", message.text)
    documents = await PlaywrightURLLoader(urls=[message.text], remove_selectors=["header", "footer"]).aload()

    for doc in documents:
        metadata = doc.metadata | {
            "source": "telegram",
            "message_sent_on": str(message.date),
        }
        ingest_internal(str(update.effective_user.id), str(message.message_id), doc.page_content, metadata)

    await message.reply_text(
        f'Url {message.text} ({message.message_id}) was successfully ingested to the user collection ({update.effective_user.id})',
        reply_to_message_id=message.message_id,
    )


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

    await message.reply_text(
        f'Document {update.message.document.file_name} ({message.message_id}) was successfully ingested to the user collection ({update.effective_user.id})',
        reply_to_message_id=message.message_id,
    )


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

app.run_polling()
