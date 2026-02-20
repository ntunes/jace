"""Tests for context compaction and memory integration in AgentCore."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jace.agent.context import ConversationContext
from jace.agent.core import AgentCore
from jace.agent.findings import FindingsTracker
from jace.agent.memory import MemoryStore
from jace.agent.metrics_store import MetricsStore
from jace.checks.registry import CheckRegistry
from jace.config.settings import LLMConfig, ScheduleConfig, Settings
from jace.device.manager import DeviceManager
from jace.llm.base import Message, Response, Role, ToolCall


def _make_settings() -> Settings:
    return Settings(
        llm=LLMConfig(provider="anthropic", model="test", api_key="k"),
        schedule=ScheduleConfig(),
    )


def _make_agent(
    *,
    llm: AsyncMock | None = None,
    memory_store: MemoryStore | None = None,
    device_manager: MagicMock | None = None,
) -> AgentCore:
    dm = device_manager or MagicMock(spec=DeviceManager)
    dm.get_connected_devices = MagicMock(return_value=["r1"])
    return AgentCore(
        settings=_make_settings(),
        llm=llm or AsyncMock(),
        device_manager=dm,
        check_registry=AsyncMock(spec=CheckRegistry),
        findings_tracker=AsyncMock(spec=FindingsTracker),
        memory_store=memory_store,
    )


# ------------------------------------------------------------------
# ConversationContext compaction
# ------------------------------------------------------------------

class TestNeedsCompaction:
    def test_below_threshold(self) -> None:
        ctx = ConversationContext(max_messages=50)
        for i in range(30):
            ctx.add_user(f"msg {i}")
        assert not ctx.needs_compaction

    def test_at_threshold(self) -> None:
        ctx = ConversationContext(max_messages=50)
        for i in range(40):
            ctx.add_user(f"msg {i}")
        assert ctx.needs_compaction

    def test_above_threshold(self) -> None:
        ctx = ConversationContext(max_messages=50)
        for i in range(45):
            ctx.add_user(f"msg {i}")
        assert ctx.needs_compaction


class TestCompact:
    def test_replaces_old_messages(self) -> None:
        ctx = ConversationContext(max_messages=50)
        for i in range(30):
            ctx.add_user(f"msg {i}")
        ctx.compact("This is a summary", keep_recent=5)
        # 5 kept + 2 synthetic summary pair
        assert len(ctx.messages) == 7

    def test_keeps_recent_messages(self) -> None:
        ctx = ConversationContext(max_messages=50)
        for i in range(20):
            ctx.add_user(f"msg {i}")
        ctx.compact("Summary text", keep_recent=5)
        # Last 5 messages should be msg 15-19
        raw_msgs = [m for m in ctx.messages if "msg" in m.content]
        assert raw_msgs[0].content == "msg 15"
        assert raw_msgs[-1].content == "msg 19"

    def test_prepends_summary_pair(self) -> None:
        ctx = ConversationContext(max_messages=50)
        ctx.add_user("hello")
        ctx.compact("A summary of events", keep_recent=10)
        msgs = ctx.messages
        assert msgs[0].role == Role.USER
        assert "summary" in msgs[0].content.lower()
        assert msgs[1].role == Role.ASSISTANT

    def test_no_summary_pair_without_compact(self) -> None:
        ctx = ConversationContext(max_messages=50)
        ctx.add_user("hello")
        msgs = ctx.messages
        assert len(msgs) == 1

    def test_clear_resets_summary(self) -> None:
        ctx = ConversationContext(max_messages=50)
        ctx.add_user("hello")
        ctx.compact("A summary", keep_recent=10)
        ctx.clear()
        assert len(ctx.messages) == 0

    def test_message_count_excludes_summary(self) -> None:
        ctx = ConversationContext(max_messages=50)
        for i in range(20):
            ctx.add_user(f"msg {i}")
        ctx.compact("Summary", keep_recent=5)
        assert ctx.message_count == 5
        assert len(ctx.messages) == 7  # 5 + 2 synthetic


# ------------------------------------------------------------------
# AgentCore._compact_context
# ------------------------------------------------------------------

class TestCompactContext:
    @pytest.mark.asyncio
    async def test_compacts_with_memory_flush(self, tmp_path: Path) -> None:
        """Full compaction: flush saves memory, summary generated, context compacted."""
        store = MemoryStore(tmp_path, max_file_size=8000, max_total_size=24000)
        store.initialize()

        # LLM returns a tool call on flush, then text on summary
        call_count = 0

        async def mock_chat(messages, tools=None, system=None, max_tokens=4096):
            nonlocal call_count
            call_count += 1
            if tools is not None:
                # Tool loop calls — first return a save_memory call, then text
                if call_count == 1:
                    return Response(
                        content="",
                        tool_calls=[ToolCall(
                            id="tc-1", name="save_memory",
                            arguments={
                                "category": "device",
                                "key": "r1",
                                "content": "Learned something",
                            },
                        )],
                    )
                return Response(content="Flush done.", stop_reason="end_turn")
            # Summary call (no tools)
            return Response(
                content="We discussed device r1 health.",
                stop_reason="end_turn",
            )

        llm = AsyncMock()
        llm.chat = AsyncMock(side_effect=mock_chat)

        agent = _make_agent(llm=llm, memory_store=store)

        # Fill context past threshold
        ctx = agent._interactive_ctx
        for i in range(45):
            ctx.add_user(f"msg {i}")

        await agent._compact_context(ctx)

        # Memory should have been saved
        assert "Learned something" in store.get_device("r1")
        # Context should be compacted
        assert ctx.message_count <= 10
        # Summary should be set
        assert "r1" in ctx.messages[0].content

    @pytest.mark.asyncio
    async def test_compacts_without_memory_store(self) -> None:
        """Compaction works even when memory_store is None."""
        llm = AsyncMock()
        llm.chat = AsyncMock(
            return_value=Response(content="Summary text.", stop_reason="end_turn"),
        )

        agent = _make_agent(llm=llm, memory_store=None)

        ctx = agent._interactive_ctx
        for i in range(45):
            ctx.add_user(f"msg {i}")

        await agent._compact_context(ctx)
        assert ctx.message_count <= 10

    @pytest.mark.asyncio
    async def test_graceful_on_llm_failure(self, tmp_path: Path) -> None:
        """If LLM calls fail, compaction still proceeds with fallback summary."""
        store = MemoryStore(tmp_path, max_file_size=8000, max_total_size=24000)
        store.initialize()

        llm = AsyncMock()
        llm.chat = AsyncMock(side_effect=Exception("LLM down"))

        agent = _make_agent(llm=llm, memory_store=store)

        ctx = agent._interactive_ctx
        for i in range(45):
            ctx.add_user(f"msg {i}")

        await agent._compact_context(ctx)
        # Should still compact
        assert ctx.message_count <= 10
        # Fallback summary
        assert "failed" in ctx.messages[0].content.lower()


# ------------------------------------------------------------------
# Tool execution: save_memory / read_memory
# ------------------------------------------------------------------

class TestMemoryToolExecution:
    @pytest.mark.asyncio
    async def test_save_memory_device(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.initialize()
        agent = _make_agent(memory_store=store)

        result = await agent._execute_tool(ToolCall(
            id="t1", name="save_memory",
            arguments={"category": "device", "key": "r1", "content": "Uses OSPF"},
        ))
        assert "Saved" in result
        assert "Uses OSPF" in store.get_device("r1")

    @pytest.mark.asyncio
    async def test_save_memory_user(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.initialize()
        agent = _make_agent(memory_store=store)

        result = await agent._execute_tool(ToolCall(
            id="t2", name="save_memory",
            arguments={"category": "user", "key": "", "content": "Prefer brief output"},
        ))
        assert "Saved" in result

    @pytest.mark.asyncio
    async def test_save_memory_incident(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.initialize()
        agent = _make_agent(memory_store=store)

        result = await agent._execute_tool(ToolCall(
            id="t3", name="save_memory",
            arguments={"category": "incident", "key": "outage-01", "content": "Root cause: bad optic"},
        ))
        assert "Saved" in result

    @pytest.mark.asyncio
    async def test_read_memory_device(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.initialize()
        store.save_device("r1", "Known OSPF quirk")
        agent = _make_agent(memory_store=store)

        result = await agent._execute_tool(ToolCall(
            id="t4", name="read_memory",
            arguments={"category": "device", "key": "r1"},
        ))
        assert "OSPF quirk" in result

    @pytest.mark.asyncio
    async def test_read_memory_listing(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.initialize()
        store.save_device("r1", "note")
        store.save_device("r2", "note")
        agent = _make_agent(memory_store=store)

        result = await agent._execute_tool(ToolCall(
            id="t5", name="read_memory",
            arguments={"category": "device"},
        ))
        assert "r1" in result
        assert "r2" in result

    @pytest.mark.asyncio
    async def test_memory_tools_without_store(self) -> None:
        agent = _make_agent(memory_store=None)

        result = await agent._execute_tool(ToolCall(
            id="t6", name="save_memory",
            arguments={"category": "device", "key": "r1", "content": "x"},
        ))
        assert "not configured" in result

        result = await agent._execute_tool(ToolCall(
            id="t7", name="read_memory",
            arguments={"category": "device"},
        ))
        assert "not configured" in result


# ------------------------------------------------------------------
# _build_system_prompt
# ------------------------------------------------------------------

class TestBuildSystemPrompt:
    def test_includes_memory_context(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.initialize()
        store.save_user_preferences("Always show set format")

        agent = _make_agent(memory_store=store)
        prompt = agent._build_system_prompt()
        assert "Always show set format" in prompt
        assert "Persistent Memory" in prompt

    def test_no_memory_store(self) -> None:
        agent = _make_agent(memory_store=None)
        prompt = agent._build_system_prompt()
        assert "Persistent Memory" not in prompt

    def test_empty_memory(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.initialize()
        agent = _make_agent(memory_store=store)
        prompt = agent._build_system_prompt()
        assert "Persistent Memory" not in prompt
