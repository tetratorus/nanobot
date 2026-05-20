# Dream memory consolidation for long-term memory.

## Your job

Read the conversation history and current memory files below. Identify the most important new facts to add and any stale facts to remove.

## Output format

Use these markers. Only include files that actually need changes.

**Add content:**
```
[FILE] <filename>
<concise bullet of the new fact to add>
```

**Remove stale content:**
```
[FILE-REMOVE] <filename>
<exact text of the stale fact to remove>
```

**Create a new skill:**
```
[SKILL] <name>
<brief description of what the skill does and when to use it>
```

If nothing needs updating, return exactly: `No changes needed.`

## Rules

1. **Top 3-5 facts only.** Don't list everything — focus on the highest-value new information.
2. **Keep bullets concise.** 1-2 sentences max per fact.
3. **Skip duplicates.** Don't add facts already present in memory files.
4. **Prioritize:** user preferences (USER.md) > gotchas (MEMORY.md) > role changes (SOUL.md).
5. **Flag contradictions.** If a memory fact is outdated, mark it [FILE-REMOVE] with the exact text.
6. **No diff syntax.** Use plain [FILE] / [FILE-REMOVE] markers. Phase 2 will apply the edits.