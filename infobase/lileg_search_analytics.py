import logging.handlers
import os

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from infobase.shared import ChatOpenRouter

file_handler = logging.handlers.RotatingFileHandler(f".logs/{os.path.basename(__file__)}.log", backupCount=10)
file_handler.doRollover()

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, logging.StreamHandler()]
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

CHROMA_CLIENT = chromadb.PersistentClient(".chroma")
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

collection_name = str(213260575)
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=EMBEDDINGS,
    client=CHROMA_CLIENT
)
logger.info("Number of documents: %s", len(vector_store.get()['documents']))

prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context} 
Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

logger.info("Prompt template: %s", prompt_template)

question = "What is a chunking strategy?"
retrieved_docs = vector_store.similarity_search(question, filter={"source": "telegram"})
logger.info("Question: %s", question)

docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
messages = prompt.invoke({"question": question, "context": docs_content})
logger.info("Messages: %s", messages)

llm = ChatOpenRouter(
    model="deepseek/deepseek-chat-v3-0324:free",
)
response = llm.invoke(messages)
answer = response.content

logger.info("Answer: %s", answer)
