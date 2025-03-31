import logging
import os
from typing import Optional
from typing import TypedDict

from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()


class GlobalState(TypedDict):
    sessions: dict[str, InMemoryChatMessageHistory]


class QuestionState(MessagesState):
    question: str
    context: list[Document]


class AnswerState(TypedDict):
    answer: str


class ChatOpenRouter(ChatOpenAI):
    def __init__(self,
                 model: str,
                 api_key: Optional[str] = None,
                 base_url: str = "https://openrouter.ai/api/v1",
                 **kwargs):
        api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        super().__init__(base_url=base_url, api_key=api_key, model=model, **kwargs)


llm = ChatOpenRouter(
    model="deepseek/deepseek-chat-v3-0324:free",
    temperature=0.7,
    max_completion_tokens=1024,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks."),
    ("system", "Use the following pieces of retrieved context to answer the question."),
    ("system", "If you don't know the answer, just say that you don't know."),
    ("system", "Use five sentences maximum and keep the answer concise."),
    ("placeholder", "{context}"),
    ("user", "{question}")
])

global_state = GlobalState(sessions={})


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in global_state["sessions"]:
        global_state["sessions"][session_id] = InMemoryChatMessageHistory()

    return global_state["sessions"][session_id]


chain = RunnableWithMessageHistory(
    prompt | llm,
    get_by_session_id,
    input_messages_key="question",
    history_messages_key="history"
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=".chroma",
)

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings, LocalFileStore(".cache"), namespace=embeddings.model_name
)

vector_store = vector_store.from_texts(["my name is oleh"], cached_embedder)


def retrieve_context(state: QuestionState):
    documents = vector_store.similarity_search(state["question"], k=12)
    return {"context": [x.page_content for x in documents]}


def chatbot(state: QuestionState, config: RunnableConfig):
    return {"messages": chain.invoke({"question": state["question"], "context": state["context"]}, config)}


builder = StateGraph(state_schema=QuestionState)
builder.add_node(retrieve_context)
builder.add_node(chatbot)
builder.add_edge(START, "retrieve_context")
builder.add_edge("retrieve_context", "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()

if __name__ == "__main__":
    graph.invoke(
        {"question": "Hello!", "messages": []},
        {"configurable": {"session_id": "1"}}
    )
    print(global_state["sessions"]["1"])

    graph.invoke(
        {"question": "Yo!", "messages": []},
        {"configurable": {"session_id": "1"}}
    )
    print(global_state["sessions"]["1"])
