"""new_empty_context tool — lets agents clear their own session."""

from __future__ import annotations

from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import (
    ContextAware,
    RequestContext,
    ToolContext,
)


class NewEmptyContextTool(Tool, ContextAware):
    """Tool that lets an agent start a fresh session.

    The cleared session's messages stay on disk in the session JSONL — Dream
    will pick them up from history.jsonl. The in-memory session is reset so
    the next turn starts with no prior conversation context.
    """

    _plugin_discoverable = True

    def __init__(self, sessions: Any):
        self._sessions = sessions
        self._channel = ""
        self._chat_id = ""

    @classmethod
    def create(cls, ctx: ToolContext) -> "NewEmptyContextTool":
        return cls(sessions=ctx.sessions)

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        return ctx.sessions is not None

    def set_context(self, ctx: RequestContext) -> None:
        self._channel = ctx.channel
        self._chat_id = ctx.chat_id

    @property
    def name(self) -> str:
        return "new_empty_context"

    @property
    def description(self) -> str:
        return (
            "Clear the current conversation context and start a fresh session. "
            "Use this when context is getting too full or cluttered. "
            "Old messages remain on disk in the session log."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    @property
    def read_only(self) -> bool:
        return False

    async def execute(self, **kwargs: Any) -> str:
        if self._sessions is None:
            return "Error: session manager unavailable."
        key = f"{self._channel}:{self._chat_id}"
        session = self._sessions.get_or_create(key)
        session.clear()
        self._sessions.save(session)
        if hasattr(self._sessions, "invalidate"):
            self._sessions.invalidate(getattr(session, "key", key))
        logger.info("Agent cleared its own context via new_empty_context (key={})", key)
        return "Context cleared. Fresh session started."
