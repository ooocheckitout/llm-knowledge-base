import logging

from sqlalchemy.orm import Session
from telegram import Update
from telegram.ext import CallbackContext

from src.common.models.message import Message
from src.common.models.review import Review, ReviewType
from src.common.models.user import User
from src.common.services.database import DatabaseService

logger = logging.getLogger(__name__)

database_service = DatabaseService()


async def keyboard_callback(update: Update, _: CallbackContext) -> None:
    query = update.callback_query

    logger.info("Processing keyboard callback %s", query.data)

    command, from_message_id = query.data.split(":")
    if command == "good" or command == "bad":
        logger.info("User %s review for message %s", query.from_user.id, from_message_id)

        with Session(database_service.engine) as session:
            user = User(
                id=query.from_user.id,
                name=query.from_user.username,
            )
            logger.info("Saving user %s into database", user)
            session.merge(user)
            session.commit()

            message = Message(
                id=int(from_message_id),
                user_id=query.from_user.id,
            )
            logger.info("Saving message %s into database", message)
            session.merge(message)
            session.commit()

            review = Review(
                user_id=query.from_user.id,
                message_id=int(from_message_id),
                feedback_type=ReviewType(command).value
            )
            logger.info("Saving review %s into database", review)
            session.merge(review)
            session.commit()

        logger.warning("Replying to command %s", command)
        await query.answer(f"User review '{command}' was saved!")
        await query.edit_message_reply_markup(reply_markup=None)
    else:
        logger.warning("Command %s is not supported!", command)
        await query.answer(f"Not supported!")
