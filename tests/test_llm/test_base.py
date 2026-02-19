"""Tests for LLM base models."""

from jace.llm.base import Message, Response, Role, ToolCall


def test_message_creation():
    msg = Message(role=Role.USER, content="hello")
    assert msg.role == Role.USER
    assert msg.content == "hello"
    assert msg.tool_calls is None


def test_response_no_tool_calls():
    resp = Response(content="answer")
    assert resp.has_tool_calls is False


def test_response_with_tool_calls():
    resp = Response(
        content="",
        tool_calls=[ToolCall(id="1", name="test", arguments={"a": 1})],
    )
    assert resp.has_tool_calls is True
    assert resp.tool_calls[0].name == "test"
