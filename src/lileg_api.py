import logging
from datetime import datetime, UTC
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect
from pydantic import BaseModel

from infobase.lileg_agent import graph, vector_store, PromptState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()


class Prompt(BaseModel):
    question: str


class FilterQuery(BaseModel):
    filter: dict[str, str]


class SearchQuery(BaseModel):
    query: str
    n_results: int = 5
    filter: Optional[dict[str, str]] = None


class Completion(BaseModel):
    answer: str


class Information(BaseModel):
    content: str
    metadata: dict[str, str]


class Identifiable(BaseModel):
    id: str


class Embedding(Identifiable):
    metadata: dict[str, str]
    content: str


@app.post("/users/{user_id}/chats/{chat_id}/complete")
async def complete(user_id: str, chat_id: str, prompt: Prompt) -> Completion:
    try:
        logger.info("Start completion %s", prompt)

        result = graph.invoke(
            PromptState(question=prompt.question, history="", context="", answer=""),
            RunnableConfig(configurable={"session_id": f"{user_id}-{chat_id}"}),
        )

        logger.info("Finish completion %s", prompt)

        return Completion(answer=result["answer"])
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


def split_documents(documents: list[Document]) -> list[Document]:
    # Telegram has a maximum message length of 4096 characters.
    # The GPT-3.5-turbo context window is 4096 tokens.
    # The token is around 4 characters.
    # We want to allow 4096 characters for a user prompt which is 1024 tokens.
    # Additionally, we can provide 3072 tokens as prompt snippets for llm context.
    # For example 12 snippents will result in 256 tokens or 1024 characters per snippet.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)


def to_documents(infos: list[Information], metadata: dict[str, str]) -> list[Document]:
    return [Document(page_content=x.content, metadata=metadata | x.metadata) for x in infos]


def safe_detect_language(text: str):
    language = "unknown"
    try:
        language = detect(text)
    except Exception as ex:
        # langdetect.lang_detect_exception.LangDetectException: No features in text.
        logger.warning("Failed to detect language! %s", ex)

    return language


@app.post("/users/{user_id}/chats/{chat_id}/remember")
async def remember(user_id: str, chat_id: str, infos: list[Information]) -> list[Identifiable]:
    try:
        logger.info("Start remembering")

        assert all(info.content != "" for info in infos)

        internal_metadata = {
            "session_id": f"{user_id}-{chat_id}",
        }
        documents = to_documents(infos, internal_metadata)
        documents = split_documents(documents)

        current_datetime_as_str = str(datetime.now(UTC))
        for document in documents:
            document.metadata["language"] = safe_detect_language(document.page_content)
            document.metadata["ingested_on"] = current_datetime_as_str

        assert all("source" in x.metadata for x in infos)
        sources = [x.metadata["source"] for x in infos]
        vector_store.delete(where={
            "$and": [
                {"session_id": f"{user_id}-{chat_id}"},
                {"source": {"$in": sources}}
            ]}
        )

        ids = await vector_store.aadd_documents(documents)

        logger.info("Finish remembering")
        return [Identifiable(id=x) for x in ids]
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/users/{user_id}/chats/{chat_id}/forgetAll")
async def forget(user_id: str, chat_id: str):
    try:
        logger.info("Start forgetting all")

        vector_store.delete(where={"session_id": f"{user_id}-{chat_id}"})

        logger.info("Finish forgetting all")
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/users/{user_id}/chats/{chat_id}/forget")
async def forget(user_id: str, chat_id: str, query: FilterQuery):
    try:
        logger.info("Start forgetting")

        vector_store.delete(where={"$and": [{"session_id": f"{user_id}-{chat_id}"}, query.filter]})

        logger.info("Finish forgetting")
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/users/{user_id}/chats/{chat_id}/similarity")
async def similarity(user_id: str, chat_id: str, query: SearchQuery) -> list[Embedding]:
    try:
        logger.info("Start similarity search")

        effective_filter = {"session_id": f"{user_id}-{chat_id}"}

        if query.filter:
            effective_filter = {"$and": [effective_filter, query.filter]}

        documents = await vector_store.asimilarity_search(
            query=query.query, k=query.n_results, filter=effective_filter
        )

        logger.info("Finish similarity search")
        return [Embedding(id=x.id, metadata=x.metadata, content=x.page_content) for x in documents]
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
