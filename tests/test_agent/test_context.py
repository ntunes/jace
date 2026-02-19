"""Tests for conversation context."""

from jace.agent.context import ConversationContext
from jace.llm.base import Message, Role


def test_context_add_messages():
    ctx = ConversationContext()
    ctx.add_user("hello")
    ctx.add_assistant(Message(role=Role.ASSISTANT, content="hi"))
    assert len(ctx.messages) == 2


def test_context_trim():
    ctx = ConversationContext(max_messages=3)
    for i in range(5):
        ctx.add_user(f"message {i}")
    assert len(ctx.messages) == 3
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
