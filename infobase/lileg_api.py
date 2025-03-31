import logging

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from infobase.lileg_agent import graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()


class Prompt(BaseModel):
    question: str


class Completion(BaseModel):
    answer: str


@app.post("/users/{user_id}/chats/{chat_id}/complete")
async def complete(user_id: str, chat_id: str, prompt: Prompt) -> Completion:
    try:
        logger.info("Start completion %s", prompt)
        result = graph.invoke(
            {"question": prompt.question},
            {"configurable": {"session_id": f"{user_id}-{chat_id}"}}
        )
        logger.info("Finish completion %s", prompt)
        return Completion(answer=result["messages"][-1].content)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
