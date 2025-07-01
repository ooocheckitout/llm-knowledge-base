from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    id: Mapped[int] = mapped_column(primary_key=True)

    def __repr__(self):
        attributes = []
        for attribute, value in self.__dict__.items():
            if not attribute.startswith("_"):
                attributes.append(f"{attribute}={value}")

        return f"{self.__class__.__name__}({', '.join(attributes)})"
