import logging

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from infobase.lileg_agent import graph

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

app = FastAPI()


class Question(BaseModel):
    question: str
    session_id: str


class Answer(BaseModel):
    answer: str


@app.post("/ask")
async def ask_question(question: Question):
    try:
        result = graph.invoke(
            {"messages": [question.question]},
            {"configurable": {"session_id": question.session_id}}
        )
        return result["messages"][-1]
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
