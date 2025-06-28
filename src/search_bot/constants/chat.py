import logging
import os
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

logger.info(f"Initializing ChatOllama({os.getenv('OLLAMA_CHAT_MODEL')})")
llm = ChatOllama(
    model=os.getenv('OLLAMA_CHAT_MODEL'),
    base_url=os.getenv('OLLAMA_BASE_URL'),
    # Settings below are required for deterministic responses
    temperature=0,
    # num_ctx=2048,
    # num_keep=0
)


class ChatOpenRouter(ChatOpenAI):
    def __init__(self,
                 model: str,
                 api_key: Optional[str] = None,
                 base_url: str = "https://openrouter.ai/api/v1",
                 **kwargs):
        api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        super().__init__(base_url=base_url, api_key=api_key, model=model, **kwargs)

# logger.info("Initializing ChatOpenRouter")
# llm = ChatOpenRouter(
#     model="deepseek/deepseek-chat-v3-0324:free",
#     temperature=0.3,
#     max_completion_tokens=1024,
# )
