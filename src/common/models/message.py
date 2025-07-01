from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from src.common.models.base import Base


class Message(Base):
    __tablename__ = "messages"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
