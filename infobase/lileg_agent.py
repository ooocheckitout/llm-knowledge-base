import logging
import os
from typing import Optional
from typing import TypedDict

from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.globals import set_llm_cache
from langchain.storage import LocalFileStore
from langchain_chroma import Chroma
from langchain_community.cache import SQLiteCache
from langchain_community.llms.fake import FakeListLLM
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.constants import START
from langgraph.graph import StateGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class GlobalState(TypedDict):
    sessions: dict[str, InMemoryChatMessageHistory]


class PromptState(TypedDict):
    question: str
    history: str
    context: str
    answer: str


class ChatOpenRouter(ChatOpenAI):
    def __init__(self,
                 model: str,
                 api_key: Optional[str] = None,
                 base_url: str = "https://openrouter.ai/api/v1",
                 **kwargs):
        api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        super().__init__(base_url=base_url, api_key=api_key, model=model, **kwargs)


llm = FakeListLLM(responses=["YO!"])
llm = ChatOpenRouter(
    model="deepseek/deepseek-chat-v3-0324:free",
    temperature=0.3,
    max_completion_tokens=1024,
)

# llm = HuggingFaceEndpoint(
#     model="openai-community/gpt2",
#     temperature=0.7,
# )

global_state = GlobalState(sessions={})


def get_message_history_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in global_state["sessions"]:
        global_state["sessions"][session_id] = InMemoryChatMessageHistory()

    return global_state["sessions"][session_id]


template = """
You are a helpful assistant.
Combine the chat history and follow up question into a standalone question:
History: {history}
Question: {question}
Answer the standalone question based on the context.
Context: {context}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | llm

# embeddings = HuggingFaceEndpointEmbeddings(
#     model="sentence-transformers/all-MiniLM-L6-v2",
#     task="feature-extraction",
# )

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings, LocalFileStore(".cached_embeddings"), namespace=embeddings.model_name
)

vector_store = Chroma(
    embedding_function=cached_embedder,
    persist_directory=".chroma",
)


def enrich_history(state: PromptState, config: RunnableConfig):
    print(enrich_history.__name__, state)

    session_id = config["configurable"]["session_id"]
    session_history = get_message_history_by_session_id(session_id)

    history = "\n".join([x.content for x in session_history.messages])

    return {"history": history}


def enrich_context(state: PromptState, config: RunnableConfig):
    print(enrich_context.__name__, state)

    documents = vector_store.similarity_search(
        query=state["question"],
        k=12,
        filter={"session_id": config["configurable"]["session_id"]},
    )
    context = "\n".join([x.page_content for x in documents])
    return {"context": context}


def chatbot(state: PromptState, config: RunnableConfig):
    print(chatbot.__name__, state)

    completion = chain.invoke({**state}, config)

    return {"answer": completion.content}


def save_history(state: PromptState, config: RunnableConfig):
    print(save_history.__name__, state)

    session_id = config["configurable"]["session_id"]
    history = get_message_history_by_session_id(session_id)

    history.add_user_message(state["question"])
    history.add_ai_message(state["answer"])

    return {}


builder = StateGraph(state_schema=PromptState)
builder.add_edge(START, "enrich_history")
builder.add_sequence([enrich_history, enrich_context, chatbot, save_history])
graph = builder.compile()

if __name__ == "__main__":
    set_llm_cache(SQLiteCache(database_path=".cached_completions"))

    graph.invoke(
        PromptState(question="My name is Oleh.", history="", context="", answer=""),
        RunnableConfig(configurable={"session_id": "1"}),
    )
    print(global_state["sessions"]["1"])

    vector_store = vector_store.from_texts(["My surname is Solomoichenko"], cached_embedder)

    graph.invoke(
        PromptState(question="What is my full name?", history="", context="", answer=""),
        RunnableConfig(configurable={"session_id": "1"}),
    )
    print(global_state["sessions"]["1"])

set_llm_cache(SQLiteCache(database_path=".cached_completions"))
