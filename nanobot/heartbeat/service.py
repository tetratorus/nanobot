"""Heartbeat service - periodic agent wake-up based on recent llmproxy activity.

Matron redesign: the heartbeat no longer reads a writable HEARTBEAT.md scratchpad.
Instead it queries llmproxy for the agent's recent request/response history, asks
an LLM to render a yes/no wake decision, and either:

  - publishes a synthetic inbound message into the agent's *real* session (the
    on_deliver path — keeps full context, no shadow session), or
  - falls back to the legacy on_execute/on_notify path (shadow session at
    session_key="heartbeat") for backwards compatibility.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

from loguru import logger

from nanobot.providers.base import LLMProvider
from nanobot.utils.llm_runtime import LLMRuntimeResolver, static_llm_runtime
from nanobot.utils.prompt_templates import render_template


class HeartbeatService:
    """Periodic heartbeat with llmproxy-aware wake decision and synthetic-inbound delivery."""

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider | None = None,
        model: str | None = None,
        on_execute: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        on_notify: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        on_deliver: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        interval_s: int = 30 * 60,
        enabled: bool = True,
        timezone: str | None = None,
        llm_runtime: LLMRuntimeResolver | None = None,
    ):
        self.workspace = workspace
        if llm_runtime is None:
            if provider is None or model is None:
                raise ValueError("HeartbeatService requires either llm_runtime or provider/model")
            llm_runtime = static_llm_runtime(provider, model)
        self._llm_runtime = llm_runtime
        self.on_execute = on_execute
        self.on_notify = on_notify
        self.on_deliver = on_deliver
        self.interval_s = interval_s
        self.enabled = enabled
        self.timezone = timezone
        self._running = False
        self._task: asyncio.Task | None = None

    # --- bot-identity + llmproxy lookup -------------------------------------

    def _get_bot_username(self) -> str | None:
        """Resolve the bot username for llmproxy queries from the agent's nanobot.json.

        Matron deployment layout: /mnt/HC_Volume_105481184/matron/agents/<name>/nanobot.json
        where <name> matches the workspace directory name.
        """
        try:
            agent_name = self.workspace.name
            config_path = Path(
                f"/mnt/HC_Volume_105481184/matron/agents/{agent_name}/nanobot.json"
            )
            if not config_path.exists():
                # Fallback: nanobot.json adjacent to the workspace
                config_path = self.workspace.parent / agent_name / "nanobot.json"
                if not config_path.exists():
                    return None
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            defaults = config.get("agents", {}).get("defaults", {})
            provider_name = defaults.get("provider", "")
            provider = config.get("providers", {}).get(provider_name, {})
            api_base = provider.get("api_base", "")
            match = re.search(r"/([^/]+_bot)/", api_base)
            return match.group(1) if match else None
        except Exception:
            return None

    def _get_recent_activity(self, bot_username: str) -> str:
        """Query llmproxy for last 1 hour OR last 10 turns (whichever has more)."""
        db_path = os.environ.get("LLMPROXY_DB", "/home/lentan/llmproxy/requests.db")
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            one_hour_ago = (
                datetime.now(timezone.utc) - timedelta(hours=1)
            ).strftime("%Y-%m-%d %H:%M:%S")
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

            rows = hour_rows if len(hour_rows) >= len(turn_rows) else turn_rows
            if not rows:
                return "No recent activity found."

            parts = []
            for row in rows:
                ts = row["timestamp"]
                body = (row["body"] or "")[:500]
                response = (row["response"] or "")[:500]
                parts.append(
                    f"[{ts}]\nRequest:\n{body}\n\nResponse:\n{response}\n"
                )
            return "\n".join(parts)
        except Exception as e:
            logger.error("Failed to query llmproxy: {}", e)
            return "Unable to query recent activity."

    # --- decision -----------------------------------------------------------

    async def _decide(self) -> tuple[str, str]:
        """Phase 1: query llmproxy + ask LLM yes/no.

        Returns (action, reason) where action is "yes" | "no".
        """
        bot_username = self._get_bot_username()
        if not bot_username:
            logger.warning("Could not determine bot username; defaulting to wake")
            return "yes", "bot username unknown"

        activity = self._get_recent_activity(bot_username)

        prompt = render_template(
            "agent/heartbeat_wake_prompt.md",
            strip=True,
            activity=activity,
        )

        llm = self._llm_runtime()
        response = await llm.provider.chat_with_retry(
            messages=[
                {"role": "system", "content": "You are a heartbeat controller. Reply ONLY 'yes' or 'no'."},
                {"role": "user", "content": prompt},
            ],
            model=llm.model,
            # Reasoning models burn the entire token budget before emitting
            # the one-word answer — disable thinking for this gate.
            reasoning_effort="minimal",
        )

        text = (response.content or "").strip().lower()
        activity_chars = len(activity)
        if "yes" in text:
            return "yes", f"agent has pending tasks (activity={activity_chars} chars)"
        return "no", f"agent idle (activity={activity_chars} chars)"

    # --- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
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
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.interval_s)
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Heartbeat error")

    async def _tick(self) -> None:
        """Single heartbeat tick: decide, then deliver (preferred) or execute."""
        action, reason = await self._decide()

        if action != "yes":
            logger.debug("Heartbeat: agent idle — skipping ({})", reason)
            return

        logger.info("Heartbeat: agent has pending tasks — waking ({})", reason)
        try:
            if self.on_deliver:
                # Preferred path: synthetic inbound into the agent's real session.
                # Full context preserved, no shadow session.
                await self.on_deliver(
                    f"[heartbeat] You have pending work. {reason}"
                )
            elif self.on_execute:
                # Legacy path: shadow session at session_key="heartbeat".
                # Kept for backwards compat; on_deliver is strongly preferred.
                response = await self.on_execute(reason)
                if response and self.on_notify:
                    await self.on_notify(response)
        except Exception:
            logger.exception("Heartbeat execution failed")

    async def trigger_now(self) -> str | None:
        """Manually trigger a heartbeat decision + execution. Returns text reply if any."""
        action, reason = await self._decide()
        if action != "yes":
            return None
        if self.on_deliver:
            await self.on_deliver(
                f"[heartbeat] You have pending work. {reason}"
            )
            return None
        if self.on_execute:
            return await self.on_execute(reason)
        return None
