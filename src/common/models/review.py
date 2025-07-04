from enum import StrEnum

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from src.common.models.base import Base


class ReviewType(StrEnum):
    positive = 'good'
    negative = 'bad'


class Review(Base):
    __tablename__ = "reviews"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"))
    feedback_type: Mapped[str] = mapped_column()
