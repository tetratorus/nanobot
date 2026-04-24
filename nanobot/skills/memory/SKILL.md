---
name: memory
description: Long-term memory you own and edit directly; Dream consolidates in the background.
always: true
---

# Memory

## Structure

- `SOUL.md` — Your role, principles, communication style. You can edit directly.
- `USER.md` — Facts about the user you serve. You can edit directly.
- `memory/MEMORY.md` — Long-term notes: gotchas, conventions, decisions. You can edit directly.
- `memory/history.jsonl` — append-only JSONL, not loaded into context. Prefer built-in `grep` to search it.

## You own your memory

When you learn something non-obvious worth keeping past this conversation — a gotcha, a convention, a policy decision — edit `memory/MEMORY.md` directly using `edit_file` or `write_file`. Keep entries terse; separate with `§` on its own line. The same applies to `SOUL.md` and `USER.md` when your role or what you know about the user changes.

Dream runs periodically as a background consolidation pass. It reads the current file before editing, so it does not clobber your direct writes. It will add new facts it notices in history, remove stale ones, and tidy duplicates — but the file remains yours.

Users can view Dream's activity with the `/dream-log` command.

## Search past events

`memory/history.jsonl` is JSONL format — each line is a JSON object with `cursor`, `timestamp`, `content`.

- For broad searches, start with `grep(..., path="memory", glob="*.jsonl", output_mode="count")` or the default `files_with_matches` mode before expanding to full content
- Use `output_mode="content"` plus `context_before` / `context_after` when you need the exact matching lines
- Use `fixed_strings=true` for literal timestamps or JSON fragments
- Use `head_limit` / `offset` to page through long histories
- Use `exec` only as a last-resort fallback when the built-in search cannot express what you need

Examples (replace `keyword`):
- `grep(pattern="keyword", path="memory/history.jsonl", case_insensitive=true)`
- `grep(pattern="2026-04-02 10:00", path="memory/history.jsonl", fixed_strings=true)`
- `grep(pattern="keyword", path="memory", glob="*.jsonl", output_mode="count", case_insensitive=true)`
- `grep(pattern="oauth|token", path="memory", glob="*.jsonl", output_mode="content", case_insensitive=true)`
