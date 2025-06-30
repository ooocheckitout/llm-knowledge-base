import logging

from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


async def echo(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("echo {} ({})".format(update.effective_user.full_name, update.effective_user.id))
    await update.effective_message.reply_text(
        f"Received a message {update.effective_message.message_id}!\n\n{update.effective_message.text}",
        reply_to_message_id=update.effective_message.message_id
    )
