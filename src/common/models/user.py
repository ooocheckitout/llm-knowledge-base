from sqlalchemy.orm import Mapped, mapped_column

from src.common.models.base import Base


class User(Base):
    __tablename__ = "users"

    name: Mapped[str] = mapped_column()