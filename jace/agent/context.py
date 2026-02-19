"""Device state and conversation context management."""

from __future__ import annotations

from jace.llm.base import Message, Role


class ConversationContext:
    """Manages conversation history for an LLM interaction."""

    def __init__(self, max_messages: int = 50) -> None:
        self._messages: list[Message] = []
        self._max_messages = max_messages

    def add_user(self, content: str) -> None:
        self._messages.append(Message(role=Role.USER, content=content))
        self._trim()

    def add_assistant(self, message: Message) -> None:
        self._messages.append(message)
        self._trim()

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._messages.append(Message(
            role=Role.TOOL, content=content, tool_call_id=tool_call_id,
        ))

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def _trim(self) -> None:
        """Trim conversation to max_messages, keeping most recent."""
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages:]
