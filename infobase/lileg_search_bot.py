import logging.handlers
import os

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from infobase.shared import ChatOpenRouter, preview

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
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_SEARCH_BOT_TOKEN')

# DATABASE AND EMBEDDINGS
CHROMA_CLIENT = chromadb.PersistentClient(".chroma")

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

llm = ChatOpenRouter(
    model="deepseek/deepseek-chat-v3-0324:free",
    temperature=0.7
)


async def welcome(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("welcome {} ({})".format(update.effective_user.full_name, update.effective_user.id))
    await update.message.reply_text(
        "\n".join([
            f"Welcome, {update.effective_user.first_name}!",
            "Send a message to search the the database."
        ]),
        reply_to_message_id=update.message.message_id
    )


async def search(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = (update.message or update.edited_message)

    logger.info("search %s", preview(message.text))

    collection_name = str(update.effective_user.id)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=EMBEDDINGS,
        client=CHROMA_CLIENT
    )

    logger.info("Performing search for query %s", preview(message.text))
    results = vector_store.similarity_search_with_score(message.text, filter={"source": "telegram"})

    if not results:
        await message.reply_text(f'No results found 😔', reply_to_message_id=message.message_id)

    for doc, score in results:
        logger.info(f"* [SIM={score:3f}; LENGTH={len(doc.page_content)}] [{doc.metadata}]")
        await message.reply_text(doc.page_content, reply_to_message_id=message.message_id)


async def search_llm(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = (update.message or update.edited_message)

    logger.info("search %s", preview(message.text))

    collection_name = str(update.effective_user.id)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=EMBEDDINGS,
        client=CHROMA_CLIENT
    )

    logger.info("Performing search for query %s", preview(message.text))
    documents = vector_store.similarity_search(message.text, k=20, filter={"source": "telegram"})
    document_context = "\n\n".join(doc.page_content for doc in documents)

    if not document_context:
        document_context = "User have not added any context to the vector database. Tell him to add some context by messaging to @lileg\_db\_bot."

    prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context} 
Answer:
        """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    messages = prompt.invoke({"question": message.text, "context": document_context})

    logger.info("Executing llm prompt for query %s with template %s", prompt_template, preview(message.text))
    response = llm.invoke(messages)

    await message.reply_markdown(response.content, reply_to_message_id=message.message_id)


app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

app.add_handler(CommandHandler("start", welcome))
app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), search_llm))

app.run_polling()
