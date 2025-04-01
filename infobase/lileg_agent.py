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
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableWithMessageHistory, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.constants import START
from langgraph.graph import StateGraph, MessagesState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class GlobalState(TypedDict):
    sessions: dict[str, InMemoryChatMessageHistory]


class VariablesDict(TypedDict):
    question: str
    context: list[str]


class PromptState(MessagesState):
    template: str
    variables: VariablesDict


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
    temperature=0.3,
    max_completion_tokens=1024,
)
llm = FakeListLLM(responses=["YO!"])

# llm = HuggingFaceEndpoint(
#     model="openai-community/gpt2",
#     temperature=0.7,
# )

global_state = GlobalState(sessions={})


def get_message_history_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in global_state["sessions"]:
        global_state["sessions"][session_id] = InMemoryChatMessageHistory()

    return global_state["sessions"][session_id]


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
    embeddings, LocalFileStore(".cache/embeddings"), namespace=embeddings.model_name
)

vector_store = Chroma(
    embedding_function=cached_embedder,
    persist_directory=".chroma",
)


def retrieve_similarities(state: PromptState, config: RunnableConfig):
    documents = vector_store.similarity_search(
        query=state["variables"]["question"],
        k=12,
        filter={"session_id": config["configurable"]["session_id"]},
    )
    return {"context": [x.page_content for x in documents] + state["variables"]["context"]}


def trim_history(state: dict, max_length: int = 1):
    if len(state["history"]) > max_length:
        state["history"] = state["history"][:max_length]
        return state

    return state


def list_variable_parser(state: dict):
    for var_key, var_value in state.items():
        if not isinstance(var_value, list):
            continue

        if all(isinstance(x, BaseMessage) for x in var_value):
            var_value = [x.content for x in var_value]

        state[var_key] = "\n\n".join([x for x in var_value])

    return state


def chatbot_with_history(state: PromptState, config: RunnableConfig):
    prompt = ChatPromptTemplate.from_template(state["template"])
    chain_with_history = RunnableWithMessageHistory(
        RunnableLambda(trim_history) | RunnableLambda(list_variable_parser) | prompt | llm,
        get_message_history_by_session_id,
        input_messages_key="question",
        history_messages_key="history",
    )
    messages = chain_with_history.invoke({**state["variables"]}, config)
    return {"messages": messages}


builder = StateGraph(state_schema=PromptState)
builder.add_sequence([retrieve_similarities, chatbot_with_history])
builder.add_edge(START, "retrieve_similarities")
graph = builder.compile()

if __name__ == "__main__":
    vector_store = vector_store.from_texts(["Capital of Poland is Niagara."], cached_embedder)

    set_llm_cache(SQLiteCache(database_path=".cache/completions"))

    template = """
You are a helpful assistant. Answer the question based on the context.
Context: {context}
Combine the chat history and follow up question into a standalone question:
History: {history}
Question: {question}
Answer the standalone question based on the context.
"""
    graph.invoke(
        PromptState(
            template=template,
            variables=VariablesDict(
                question="What is the capital of Poland?",
                context=["Capital of Poland is Mumbai."]
            ),
            messages=[]
        ),
        RunnableConfig(configurable={"session_id": "1"}),
    )
    print(global_state["sessions"]["1"])

    graph.invoke(
        PromptState(
            template=template,
            variables=VariablesDict(question="What is the capital of Poland?", context=[]),
            messages=[]
        ),
        RunnableConfig(configurable={"session_id": "1"}),
    )
    print(global_state["sessions"]["1"])

set_llm_cache(SQLiteCache(database_path=".cache/completions"))
