import logging.handlers
import os

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from infobase.shared import preview, EMBEDDINGS, CHROMA_CLIENT_DIR, LLM

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


async def search(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("search %s", preview(message.text))

    collection_name = str(update.effective_user.id)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=EMBEDDINGS,
        persist_directory=CHROMA_CLIENT_DIR,
    )

    logger.info("Performing search for query %s", preview(message.text))
    results = vector_store.similarity_search_with_score(message.text)

    if not results:
        await message.reply_text(f'No results found 😔', reply_to_message_id=message.message_id)

    for doc, score in results:
        logger.info(f"* [SIM={score:3f}; LENGTH={len(doc.page_content)}] [{doc.metadata}]")
        await message.reply_text(doc.page_content, reply_to_message_id=message.message_id)


async def search_llm(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("search %s", preview(message.text))

    collection_name = str(update.effective_user.id)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=EMBEDDINGS,
        persist_directory=CHROMA_CLIENT_DIR,
    )

    logger.info("Performing search for query %s", preview(message.text))
    documents = vector_store.similarity_search(message.text, k=20)
    document_context = "\n\n".join(doc.page_content for doc in documents)

    if not document_context:
        document_context = (
            "User have not added any context to the vector database. Tell him to add some context by messaging to @lileg_db_bot."
        )

    prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context} 
Answer:
        """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    messages = prompt.invoke({"question": message.text, "context": document_context})

    logger.info("Executing llm prompt for query %s with template %s", preview(message.text), prompt_template)
    response = LLM.invoke(messages)

    await message.reply_text(response.content, reply_to_message_id=message.message_id)


TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_SEARCH_BOT_TOKEN')
app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

app.add_handler(CommandHandler("start", welcome))
app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), search_llm))

app.run_polling()
