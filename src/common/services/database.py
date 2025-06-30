import logging
import os

from sqlalchemy import create_engine

from src.common.models.base import Base

logger = logging.getLogger(__name__)


class DatabaseService:
    def __init__(self):
        self._engine = None

    @property
    def engine(self):
        if not self._engine:
            self._engine = self.initialize()

        return self._engine

    @staticmethod
    def initialize():
        logger.info(f"Initializing sqlalchemy")
        engine = create_engine(os.getenv('DB_CONNECTION_STRING'))

        logger.info(f"Recreating tables")
        # Using our table metadata and our engine, we can generate our schema at once in our target SQLite database.
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)

        return engine
