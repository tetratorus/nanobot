"""Heartbeat service - periodic agent wake-up to check for tasks."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from loguru import logger

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_HEARTBEAT_YESNO_PROMPT = """You are a heartbeat controller. Your job is to decide whether an AI agent should be woken up.

Review the agent's recent activity below and reply with ONLY "yes" or "no".

Rules:
- "yes" = the agent has pending tasks, unanswered questions, promised follow-ups, or is in the middle of active work
- "no" = the agent is idle, waiting, or has no pending work

Recent agent activity (last requests/responses):
{activity}

Heartbeat context:
{heartbeat_content}

Reply with ONLY "yes" or "no"."""


class HeartbeatService:
    """
    Periodic heartbeat service with smart wake.

    Phase 1 (decision): query llmproxy for recent activity, combine with HEARTBEAT.md,
    ask LLM yes/no via lightweight call.

    Phase 2 (execution): only triggered when Phase 1 returns "yes".
    Sends Telegram nudge to main session.
    """

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        on_execute: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        on_notify: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        on_deliver: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        interval_s: int = 30 * 60,
        enabled: bool = True,
        timezone: str | None = None,
    ):
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.on_execute = on_execute
        self.on_notify = on_notify
        self.on_deliver = on_deliver
        self.interval_s = interval_s
        self.enabled = enabled
        self.timezone = timezone
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def heartbeat_file(self) -> Path:
        return self.workspace / "HEARTBEAT.md"

    def _read_heartbeat_file(self) -> str | None:
        if self.heartbeat_file.exists():
            try:
                return self.heartbeat_file.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    def _get_bot_username(self) -> str | None:
        """Get bot username from nanobot.json for llmproxy queries."""
        try:
            # Derive nanobot.json path from workspace path
            agent_name = self.workspace.name
            config_path = Path(f"/mnt/HC_Volume_105481184/matron/agents/{agent_name}/nanobot.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            defaults = config.get("agents", {}).get("defaults", {})
            provider_name = defaults.get("provider", "")
            provider = config.get("providers", {}).get(provider_name, {})
            api_base = provider.get("api_base", "")
            # Match bot username in URL path
            match = re.search(r"/([^/]+_bot)/", api_base)
            return match.group(1) if match else None
        except Exception:
            return None

    def _get_recent_activity(self, bot_username: str) -> str:
        """Query llmproxy for last 1 hour OR last 10 turns (whichever is longer)."""
        db_path = os.environ.get("LLMPROXY_DB", "/home/lentan/llmproxy/requests.db")
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get last 1 hour of requests
            one_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                """
                SELECT body, response, timestamp
                FROM requests
                WHERE agent = ? AND timestamp > ?
                ORDER BY timestamp DESC
                """,
                (bot_username, one_hour_ago),
            )
            hour_rows = cursor.fetchall()

            # Get last 10 turns (regardless of time)
            cursor.execute(
                """
                SELECT body, response, timestamp
                FROM requests
                WHERE agent = ?
                ORDER BY timestamp DESC
                LIMIT 10
                """,
                (bot_username,),
            )
            turn_rows = cursor.fetchall()
            conn.close()

            # Use whichever has more content
            if len(hour_rows) >= len(turn_rows):
                rows = hour_rows
            else:
                rows = turn_rows

            if not rows:
                return "No recent activity found."

            # Build activity summary
            parts = []
            for row in rows:
                ts = row["timestamp"]
                body = row["body"] or ""
                response = row["response"] or ""
                # Truncate for brevity
                body_preview = body[:500] + "..." if len(body) > 500 else body
                resp_preview = response[:500] + "..." if len(response) > 500 else response
                parts.append(f"[{ts}]\nRequest:\n{body_preview}\n\nResponse:\n{resp_preview}\n")

            return "\n".join(parts)
        except Exception as e:
            logger.error("Failed to query llmproxy: {}", e)
            return "Unable to query recent activity."

    async def _decide(self, heartbeat_content: str) -> tuple[str, str]:
        """Phase 1: query llmproxy + HEARTBEAT.md, ask LLM yes/no.

        Returns ("yes" | "no", reason).
        """
        bot_username = self._get_bot_username()
        if not bot_username:
            logger.warning("Could not determine bot username, defaulting to wake")
            return "yes", "bot username unknown"

        activity = self._get_recent_activity(bot_username)

        prompt = _HEARTBEAT_YESNO_PROMPT.format(
            activity=activity,
            heartbeat_content=heartbeat_content,
        )

        response = await self.provider.chat_with_retry(
            messages=[
                {"role": "system", "content": "You are a heartbeat controller. Reply ONLY 'yes' or 'no'."},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
        )

        text = response.content.strip().lower()
        if "yes" in text:
            return "yes", f"agent has pending tasks (recent activity: {len(activity)} chars)"
        else:
            return "no", f"agent idle (recent activity: {len(activity)} chars)"

    async def start(self) -> None:
        """Start the heartbeat service."""
        if not self.enabled:
            logger.info("Heartbeat disabled")
            return
        if self._running:
            logger.warning("Heartbeat already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Heartbeat started (every {}s)", self.interval_s)

    def stop(self) -> None:
        """Stop the heartbeat service."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        """Main heartbeat loop."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_s)
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat error: {}", e)

    async def _tick(self) -> None:
        """Execute a single heartbeat tick."""
        content = self._read_heartbeat_file() or ""

        action, reason = await self._decide(content)

        if action == "yes":
            logger.info("Heartbeat: agent has pending tasks — waking ({})", reason)
            try:
                if self.on_deliver:
                    await self.on_deliver(f"Heartbeat woke agent. Agent has pending tasks. ({reason})")
                elif self.on_execute:
                    response = await self.on_execute(content)
                    if response and self.on_notify:
                        await self.on_notify(response)
            except Exception:
                logger.exception("Heartbeat execution failed")
        else:
            logger.debug("Heartbeat: agent idle — skipping ({})", reason)

    async def trigger_now(self) -> str | None:
        """Manually trigger a heartbeat."""
        content = self._read_heartbeat_file()
        if not content:
            return None
        action, reason = await self._decide(content)
        if action == "yes":
            if self.on_deliver:
                await self.on_deliver(f"Heartbeat woke agent. Agent has pending tasks. ({reason})")
                return "woken"
            if self.on_execute:
                return await self.on_execute(content)
        return "skipped"
