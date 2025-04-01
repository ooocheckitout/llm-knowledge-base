import logging
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from infobase.lileg_agent import graph, vector_store, PromptState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()


class Prompt(BaseModel):
    question: str


class SearchQuery(BaseModel):
    query: str
    n_results: int = 5
    filter: Optional[dict[str, str]] = {"1": "1"}


class Completion(BaseModel):
    answer: str


class Information(BaseModel):
    text: str
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


@app.post("/users/{user_id}/chats/{chat_id}/vectorize")
async def vectorize(user_id: str, chat_id: str, infos: list[Information]) -> list[Identifiable]:
    try:
        logger.info("Start vectorization")

        internal_metadata = {
            "session_id": f"{user_id}-{chat_id}",
        }
        documents = [Document(page_content=x.text, metadata=internal_metadata | x.metadata) for x in infos]

        assert all("source" in x.metadata for x in infos)
        sources = [x.metadata["source"] for x in infos]
        vector_store.delete(where={
            "$and": [
                {"session_id": f"{user_id}-{chat_id}"},
                {"source": {"$in": sources}}
            ]}
        )

        ids = await vector_store.aadd_documents(documents)

        logger.info("Finish vectorization")
        return [Identifiable(id=x) for x in ids]
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/users/{user_id}/chats/{chat_id}/similarity")
async def vectorize(user_id: str, chat_id: str, search: SearchQuery) -> list[Embedding]:
    try:
        logger.info("Start similarity search")

        combined_filter = {"session_id": f"{user_id}-{chat_id}"}

        if search.filter:
            combined_filter = {"$and": [combined_filter, search.filter]}

        documents = await vector_store.asimilarity_search(
            query=search.query, k=search.n_results, filter=combined_filter
        )

        logger.info("Finish similarity search")
        return [Embedding(id=x.id, metadata=x.metadata, content=x.page_content) for x in documents]
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
