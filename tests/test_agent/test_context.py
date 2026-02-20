"""Tests for conversation context."""

from jace.agent.context import ConversationContext
from jace.llm.base import Message, Role


def test_context_add_messages():
    ctx = ConversationContext()
    ctx.add_user("hello")
    ctx.add_assistant(Message(role=Role.ASSISTANT, content="hi"))
    assert len(ctx.messages) == 2


def test_context_messages_accumulate():
    """Messages accumulate without automatic trimming (compaction handles that)."""
    ctx = ConversationContext(max_messages=3)
    for i in range(5):
        ctx.add_user(f"message {i}")
    assert ctx.message_count == 5


def test_context_safety_trim():
    """_trim() still works as a manual safety net."""
    ctx = ConversationContext(max_messages=3)
    for i in range(5):
        ctx.add_user(f"message {i}")
    ctx._trim()
    assert ctx.message_count == 3
    assert ctx.messages[0].content == "message 2"


def test_context_clear():
    ctx = ConversationContext()
    ctx.add_user("test")
    ctx.clear()
    assert len(ctx.messages) == 0


def test_context_tool_result():
    ctx = ConversationContext()
    ctx.add_tool_result("call-1", "result data")
    assert ctx.messages[0].role == Role.TOOL
    assert ctx.messages[0].tool_call_id == "call-1"


def test_message_count_property():
    ctx = ConversationContext()
    ctx.add_user("a")
    ctx.add_assistant(Message(role=Role.ASSISTANT, content="b"))
    assert ctx.message_count == 2


def test_needs_compaction_threshold():
    ctx = ConversationContext(max_messages=10)
    for i in range(7):
        ctx.add_user(f"msg {i}")
    assert not ctx.needs_compaction
    ctx.add_user("msg 7")
    assert ctx.needs_compaction  # 8 = 80% of 10
