import logging
import os
from datetime import datetime, UTC
from pathlib import Path

from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader, PlaywrightURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, Message
from telegram import Update
from telegram.ext import ContextTypes

from src.common.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)
embedding_service = EmbeddingService()


async def reply_message(message: Message):
    keyboard_markup = InlineKeyboardMarkup([
        [InlineKeyboardButton("Delete", callback_data=f"delete:{message.message_id}"), ],
    ])

    await message.reply_text(
        f'Message ({message.message_id}) was successfully ingested to the user collection ({message.chat_id})',
        reply_to_message_id=message.message_id,
        reply_markup=keyboard_markup
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


async def ingest_internal(message: Message, documents: list[Document]):
    documents = list(filter(lambda doc: doc.page_content != "", documents))

    if not documents:
        await message.reply_text(f'Nothing to ingest 😔', reply_to_message_id=message.message_id)
        return

    message_id_as_str = str(message.message_id)

    logger.info("Splitting %s documents for message id %s", len(documents), message_id_as_str)
    documents = split_documents(documents)

    current_datetime_as_str = str(datetime.now(UTC))
    for document in documents:
        document.metadata["message_id"] = message_id_as_str
        document.metadata["message_date"] = str(message.date)

        document.metadata["language"] = safe_detect_language(document.page_content)
        document.metadata["ingested_on"] = current_datetime_as_str

    vector_store = Chroma(
        collection_name=str(message.chat_id),
        embedding_function=embedding_service.embedding,
        persist_directory=os.getenv('CHROMA_PERSIST_DIR'),
    )

    logger.info("Removing existing documents in %s for message id %s", message.chat_id, message_id_as_str)
    vector_store.delete(where={"message_id": message_id_as_str})

    logger.info(
        "Writing %s documents in %s for message id %s", len(documents), message.chat_id, message_id_as_str
    )
    await vector_store.aadd_documents(documents)

    for document in documents:
        await message.reply_text(document.page_content, reply_to_message_id=message.message_id)


async def ingest_text(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("Start indexing text from message id %s", message.message_id)

    documents = [
        Document(page_content=message.text, metadata={"source": str(message.message_id)})
    ]
    [x.metadata.update({"source_type": "message"}) for x in documents]
    await ingest_internal(message, documents)

    logger.info("Finish indexing text from message id %s", message.message_id)

    await reply_message(message)


async def ingest_url(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("Start indexing url from message id %s", message.message_id)

    logger.info("Loading website url from message id %s", message.message_id)
    documents = await PlaywrightURLLoader(
        urls=message.text.splitlines(), remove_selectors=["header", "footer"]
    ).aload()

    for url in message.text.splitlines():
        try:
            logger.info("Loading youtube url from message id %s", message.message_id)
            youtube_documents = await YoutubeLoader.from_youtube_url(
                youtube_url=url, language=["en", "ru"]
            ).aload()

            documents.extend(youtube_documents)
        except Exception as ex:
            # transcription not available
            logger.warning("Failed to load youtube url %s, %s", url, ex)

    logger.info("Enhancing metadata url from message id %s", message.message_id)
    [x.metadata.update({"source_type": "url"}) for x in documents]
    await ingest_internal(message, documents)

    logger.info("Finish indexing url from message id %s", message.message_id)

    await reply_message(message)


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

    [x.metadata.update({"source_type": "file"}) for x in documents]
    await ingest_internal(message, documents)

    logger.info("Finish indexing file from message id %s", message.message_id)

    await reply_message(message)
