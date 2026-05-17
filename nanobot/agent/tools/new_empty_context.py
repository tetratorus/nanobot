"""new_empty_context tool — lets agents clear their own session."""

from __future__ import annotations

from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


class NewEmptyContextTool(Tool):
    """Tool that lets an agent start a fresh session, archiving the old one."""

    def __init__(self, loop):
        self._loop = loop
        self._channel = ""
        self._chat_id = ""

    def set_context(self, channel: str, chat_id: str) -> None:
        self._channel = channel
        self._chat_id = chat_id

    @property
    def name(self) -> str:
        return "new_empty_context"

    @property
    def description(self) -> str:
        return (
            "Clear the current conversation context and start a fresh session. "
            "Archives the old session so no information is lost. "
            "Use this when context is getting too full or cluttered."
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
        key = f"{self._channel}:{self._chat_id}"
        session = self._loop.sessions.get_or_create(key)
        snapshot = session.messages[session.last_consolidated:]
        session.clear()
        self._loop.sessions.save(session)
        self._loop.sessions.invalidate(session.key)
        if snapshot:
            self._loop._schedule_background(self._loop.consolidator.archive(snapshot))
        logger.info("Agent cleared its own context via new_empty_context tool (key={})", key)
        return "Context cleared. Fresh session started. Old session has been archived."
