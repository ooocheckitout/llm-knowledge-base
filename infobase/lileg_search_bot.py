import logging.handlers
import os

import telegramify_markdown
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters, CallbackContext, \
    CallbackQueryHandler

from infobase.lileg_agent import cached_embedder, prompt, llm, get_message_history_by_session_id

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


async def similarity_search(collection_name: str, query: str, n_results: int) -> list[Document]:
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=cached_embedder,
        persist_directory=".chroma",
    )
    return await vector_store.asimilarity_search(query=query, k=n_results)


async def search(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("Searching for message id %s", message.message_id)
    documents = await similarity_search(str(message.chat_id), message.text, 3)

    logger.info("Replying for message id %s", message.message_id)
    if not any(documents):
        await message.reply_text(f'No results found 😔', reply_to_message_id=message.message_id)

    for document in documents:
        await message.reply_text(document.page_content, reply_to_message_id=message.message_id)


async def search_llm(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    logger.info("Searching for message id %s", message.message_id)

    documents = await similarity_search(str(message.chat_id), message.text, 12)

    logger.info("Enriching context for message id %s", message.message_id)

    context_template = """
The source for the following context is {source_type} {source}:
"{content}" 
        """

    context = "\n".join([
        context_template
        .replace("{content}", x.page_content)
        .replace("{source_type}", x.metadata["source_type"])
        .replace("{source}", x.metadata["source"])
        for x in documents
    ])
    missing_context = r"No context is available. Try adding more information to @lileg_db_bot."

    logger.info("Enriching history for message id %s", message.message_id)
    session_history = get_message_history_by_session_id(str(message.chat_id))
    history = "\n".join([f"{x.type}: \"{x.content}\"" for x in session_history.messages])

    logger.info("Prompting for message id %s", message.message_id)
    chain = prompt | llm
    completion = chain.invoke({"question": message.text, "history": history, "context": context or missing_context})

    logger.info("Saving history for message id %s", message.message_id)
    session_history.add_user_message(message.text)
    session_history.add_ai_message(completion.content)

    logger.info("Replying for message id %s", message.message_id)
    markdown_content = telegramify_markdown.markdownify(completion.content)

    keyboard_markup = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Good response", callback_data=f"good:{message.message_id}"),
            InlineKeyboardButton("Bad response", callback_data=f"bad:{message.message_id}"),
        ],
    ])

    await message.reply_markdown_v2(
        markdown_content,
        reply_to_message_id=message.message_id,
        reply_markup=keyboard_markup,
    )


async def keyboard_callback(update: Update, _: CallbackContext) -> None:
    query = update.callback_query

    logger.info("Received Keyboard callback %s", query.data)

    command, message_id = query.data.split(":")
    if command == "good" or command == "bad":
        logger.info("User review for message %s", message_id)

        await query.answer(f"User review '{command}' was saved!")
        await query.edit_message_reply_markup(reply_markup=None)
    else:
        await query.answer(f"Not supported!")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.warning(f'Update "{update}" caused error "{context.error}"')


TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_SEARCH_BOT_TOKEN')
app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

app.add_handler(CommandHandler("start", welcome))
app.add_handler(CommandHandler("similarity", search))
app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), search_llm))
app.add_handler(CallbackQueryHandler(keyboard_callback))
app.add_error_handler(error_handler)

app.run_polling()
