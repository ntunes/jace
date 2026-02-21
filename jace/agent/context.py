"""Device state and conversation context management."""

from __future__ import annotations

from jace.llm.base import Message, Role


class ConversationContext:
    """Manages conversation history for an LLM interaction."""

    def __init__(self, max_messages: int = 50) -> None:
        self._messages: list[Message] = []
        self._max_messages = max_messages
        self._summary: str | None = None

    @property
    def message_count(self) -> int:
        """Number of raw messages (excludes synthetic summary pair)."""
        return len(self._messages)

    @property
    def needs_compaction(self) -> bool:
        """True when messages reach 80% of capacity."""
        return len(self._messages) >= int(self._max_messages * 0.8)

    def add_user(self, content: str) -> None:
        self._messages.append(Message(role=Role.USER, content=content))

    def add_assistant(self, message: Message) -> None:
        self._messages.append(message)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._messages.append(Message(
            role=Role.TOOL, content=content, tool_call_id=tool_call_id,
        ))

    @property
    def messages(self) -> list[Message]:
        """Return messages, prepending synthetic summary pair if set."""
        result: list[Message] = []
        if self._summary:
            result.append(Message(
                role=Role.USER,
                content=f"[Previous conversation summary]: {self._summary}",
            ))
            result.append(Message(
                role=Role.ASSISTANT,
                content="Understood. I have context from our earlier conversation.",
            ))
        result.extend(self._messages)
        return result

    def compact(self, summary: str, keep_recent: int = 10) -> None:
        """Replace older messages with a summary, keeping last N messages."""
        self._summary = summary
        if len(self._messages) > keep_recent:
            self._messages = self._messages[-keep_recent:]

    @property
    def raw_messages(self) -> list[Message]:
        """Return raw messages without synthetic summary prefix."""
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()
        self._summary = None

    def _trim(self) -> None:
        """Safety-net trim — keeps most recent messages if limit exceeded."""
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages:]
