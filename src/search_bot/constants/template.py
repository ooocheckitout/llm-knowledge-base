import logging

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

template = """
User Query:
"{question}"

Retrieved Context:
{context}

Conversation History:
{history}

Response Instruction:
"Use the retrieved data to generate an accurate and contextually relevant response.
Prioritize retrieved information over general knowledge.
If multiple sources provide similar information, summarize and cite all relevant sources.
If conflicting information appears, present all perspectives naturally.
If no relevant data is found, acknowledge this and either request clarification or generate a response based on general knowledge.
Use three sentences maximum and keep the response concise, factual, and structured."

Response:
"""

logger.info("Initializing ChatPromptTemplate")
prompt = ChatPromptTemplate.from_template(template)