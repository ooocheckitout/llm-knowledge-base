import logging
import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        providers = {
            "ollama": self.initialize_ollama,
            "openai": self.initialize_openai
        }
        model_name = os.getenv('CHAT_MODEL', 'phi4-mini:3.8b')
        self._chat = providers[os.getenv('CHAT_PROVIDER', 'ollama')](model_name)

    @property
    def llm(self):
        return self._chat

    def ask(self, question: str):
        return self._chat.invoke(question)

    @staticmethod
    def initialize_ollama(model_name: str):
        logger.info(f"Initializing ChatOllama({model_name})")
        return ChatOllama(
            model=model_name,
            base_url=os.getenv('OLLAMA_BASE_URL'),
            temperature=0,
        )

    @staticmethod
    def initialize_openai(model_name: str):
        logger.info(f"Initializing ChatOpenAI({model_name})")
        return ChatOpenAI(
            model=model_name,
            base_url=os.getenv('OPENAI_BASE_URL'),
            api_key=os.getenv('OPENAI_API_KEY'),
            temperature=0,
        )
