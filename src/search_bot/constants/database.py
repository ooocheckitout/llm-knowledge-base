import logging
import os

from sqlalchemy import create_engine

from src.search_bot.models.base import Base

logger = logging.getLogger(__name__)

logger.info(f"Initializing sqlalchemy")
# The echo=True parameter indicates that SQL emitted by connections will be logged to standard out.
engine = create_engine(os.getenv('DB_CONNECTION_STRING'))

logger.info(f"Recreating all tables")
# Using our table metadata and our engine, we can generate our schema at once in our target SQLite database.
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
