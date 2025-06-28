import logging
import os
import sys
from dotenv import load_dotenv
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackQueryHandler

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger.info("Loading environment variables")
load_dotenv()

from src.search_bot.commands.welcome import welcome
from src.search_bot.handlers.keyboard import keyboard_callback
from src.search_bot.handlers.error import error_handler
from src.search_bot.commands.search import search, search_llm

logger.info("Starting application")
app = ApplicationBuilder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()

app.add_handler(CommandHandler("start", welcome))
app.add_handler(CommandHandler("search", search))
app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), search_llm))
app.add_handler(CallbackQueryHandler(keyboard_callback))
app.add_error_handler(error_handler)

logger.info("Waiting for requests")
app.run_polling()
logger.info("Shutting down application")
