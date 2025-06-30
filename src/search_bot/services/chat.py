import logging
import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        self._chat = None
        self.providers = {
            "ollama": self.initialize_ollama,
            "openai": self.initialize_openai
        }

    @property
    def llm(self):
        if not self._chat:
            self._chat = self.providers[os.getenv('CHAT_PROVIDER')]()

        return self._chat

    @staticmethod
    def initialize_ollama():
        logger.info(f"Initializing ChatOllama({os.getenv('CHAT_MODEL')})")
        return ChatOllama(
            model=os.getenv('CHAT_MODEL'),
            base_url=os.getenv('OLLAMA_BASE_URL'),
            temperature=0,
        )

    @staticmethod
    def initialize_openai():
        logger.info(f"Initializing ChatOpenAI({os.getenv('CHAT_MODEL')})")
        return ChatOpenAI(
            model=os.getenv('CHAT_MODEL'),
            base_url=os.getenv('OPENAI_BASE_URL'),
            api_key=os.getenv('OPENAI_API_KEY'),
            temperature=0,
        )
