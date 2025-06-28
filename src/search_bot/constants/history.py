import logging
from typing import TypedDict

from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory

logger = logging.getLogger(__name__)


class GlobalState(TypedDict):
    sessions: dict[str, InMemoryChatMessageHistory]


global_state = GlobalState(sessions={})


def get_message_history_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in global_state["sessions"]:
        global_state["sessions"][session_id] = InMemoryChatMessageHistory()

    return global_state["sessions"][session_id]
