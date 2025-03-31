import hashlib
import logging

import langchain.docstore.document
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from infobase.lileg_agent import graph, vector_store

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

app = FastAPI()


class Question(BaseModel):
    user_id: str
    question: str
    context: str


class Answer(BaseModel):
    answer: str


class Embedding(BaseModel):
    text: str
    metadata: dict[str, str]


class Query(BaseModel):
    text: str
    filter: dict[str, str]
    n_results: int


class Results(BaseModel):
    results: list[str]


@app.post("/llm")
async def question_llm(question: Question) -> Answer:
    try:
        logger.info("Start ask question %s", question.question)
        result = graph.invoke(
            {"question": question.question, "messages": [], "context": [question.context]},
            {"configurable": {"session_id": question.user_id}}
        )
        logger.info("Finish ask question %s", question.question)
        return Answer(answer=result["messages"][-1].content)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


def hash_text(text: str, algorithm="sha256", encoding="utf-8") -> str:
    hasher = hashlib.new(algorithm)
    hasher.update(text.encode(encoding))
    return hasher.hexdigest()


@app.post("/store")
async def store_texts(embedding: Embedding) -> list[str]:
    try:
        embedding_hash = hash_text(embedding.text)
        internal_metadata = {"hash": embedding_hash}
        document = langchain.docstore.document.Document(embedding.text, metadata=embedding.metadata | internal_metadata)
        vector_store.delete(where={"hash": embedding_hash})
        return await vector_store.aadd_documents([document])
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity")
async def similarity_search(query: Query) -> Results:
    try:
        documents = await vector_store.asimilarity_search(query.text, k=query.n_results, filter=query.filter)
        return Results(results=[x.page_content for x in documents])
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
