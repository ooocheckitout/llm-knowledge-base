import logging.handlers
import os
import sys

from dotenv import load_dotenv
from telegram import Message
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackQueryHandler

from src.common.services.embedding import EmbeddingService
from src.db_bot.commands.clear import clear
from src.db_bot.commands.ingest import ingest_url, ingest_text, ingest_file
from src.db_bot.commands.welcome import welcome
from src.db_bot.handlers.error import error_handler
from src.db_bot.handlers.keyboard import keyboard_callback

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

embedding_service = EmbeddingService()

embedding_service.embedding.embed_documents(["aaa"])


class EnsureSingleEntity(filters.Entity):
    def filter(self, message: Message) -> bool:
        return message.entities and all(entity.type == self.entity_type for entity in message.entities)


app = ApplicationBuilder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()

app.add_handler(CommandHandler("start", welcome))
app.add_handler(CommandHandler("clear", clear))
app.add_handler(MessageHandler(EnsureSingleEntity("url"), ingest_url))
app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), ingest_text))
app.add_handler(MessageHandler(filters.Document.ALL, ingest_file))
app.add_handler(CallbackQueryHandler(keyboard_callback))
app.add_error_handler(error_handler)

logger.info("Waiting for requests")
app.run_polling()
logger.info("Shutting down application")
