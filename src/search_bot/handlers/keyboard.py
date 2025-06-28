import logging

from sqlalchemy.orm import Session
from telegram import Update
from telegram.ext import CallbackContext

from src.search_bot.constants.database import engine
from src.search_bot.models.message import Message
from src.search_bot.models.review import Review, ReviewType
from src.search_bot.models.user import User

logger = logging.getLogger(__name__)


async def keyboard_callback(update: Update, _: CallbackContext) -> None:
    query = update.callback_query

    logger.info("Received keyboard callback %s", query.data)

    command, from_message_id = query.data.split(":")
    if command == "good" or command == "bad":
        logger.info("User %s review for message %s", query.from_user.id, from_message_id)

        with Session(engine) as session:
            user = User(
                id=query.from_user.id,
                name=query.from_user.username,
            )
            session.merge(user)
            session.commit()

            message = Message(
                id=int(from_message_id),
                user_id=query.from_user.id,
            )
            session.merge(message)
            session.commit()

            review = Review(
                user_id=query.from_user.id,
                message_id=int(from_message_id),
                feedback_type=ReviewType(command).value
            )
            session.merge(review)
            session.commit()

        await query.answer(f"User review '{command}' was saved!")
        await query.edit_message_reply_markup(reply_markup=None)
    else:
        await query.answer(f"Not supported!")
