import logging
import os
from typing import Optional, Annotated
from typing import TypedDict

from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.globals import set_llm_cache
from langchain.storage import LocalFileStore
from langchain_chroma import Chroma
from langchain_community.cache import SQLiteCache
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class GlobalState(TypedDict):
    sessions: dict[str, InMemoryChatMessageHistory]


class PromptState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    context: Annotated[list[AnyMessage], add_messages]
    question: str
    answer: str


class ChatOpenRouter(ChatOpenAI):
    def __init__(self,
                 model: str,
                 api_key: Optional[str] = None,
                 base_url: str = "https://openrouter.ai/api/v1",
                 **kwargs):
        api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        super().__init__(base_url=base_url, api_key=api_key, model=model, **kwargs)


# llm = ChatOpenRouter(
#     model="deepseek/deepseek-chat-v3-0324:free",
#     temperature=0.7,
#     max_completion_tokens=1024,
# )

from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    model="openai-community/gpt2",
    max_new_tokens=512,
    temperature=0.7,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks."),
    ("system", "Use the following pieces of retrieved context to answer the question."),
    ("system", "If you don't know the answer, just say that you don't know."),
    ("system", "Use five sentences maximum and keep the answer concise."),
    ("placeholder", "{context}"),
    ("user", "{question}"),
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
    history_messages_key="history",
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings, LocalFileStore(".cache/embeddings"), namespace=embeddings.model_name
)

vector_store = Chroma(
    embedding_function=cached_embedder,
    persist_directory=".chroma",
)


def retrieve_context(state: PromptState):
    documents = vector_store.similarity_search(
        query=state["question"],
        k=12,
    )
    return {"context": [x.page_content for x in documents]}


def chatbot(state: PromptState, config: RunnableConfig):
    messages = chain.invoke({"question": state["question"], "context": state["context"]}, config)
    return {"messages": messages}


builder = StateGraph(state_schema=PromptState)
builder.add_node(retrieve_context)
builder.add_node(chatbot)
builder.add_edge(START, "retrieve_context")
builder.add_edge("retrieve_context", "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()

set_llm_cache(SQLiteCache(database_path=".cache/completions"))

if __name__ == "__main__":
    vector_store = vector_store.from_texts(["Capital of Poland is Niagara."], cached_embedder)

    graph.invoke(
        {"question": "Hello! My name is Oleh."},
        {"configurable": {"session_id": "1"}}
    )
    print(global_state["sessions"]["1"])

    graph.invoke(
        {"question": "Yo! What is the capital of Poland?"},
        {"configurable": {"session_id": "1"}}
    )
    print(global_state["sessions"]["1"])
