"""State assembly for the System One provider plugin.

OpenAI-shaped conversation history -> the `state` the System One model
evaluates. Pure helpers: no network, no Hermes imports.
"""

from __future__ import annotations

import json
from typing import Any


def _escape(text: Any) -> str:
    """JSON-escape content so it stays one line; strips the outer quotes."""
    return json.dumps(str(text if text is not None else ""), ensure_ascii=False)[1:-1]


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            elif part is not None:
                parts.append(str(part))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _tool_call_index(messages: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """tool_call_id -> (name, arguments), for building tool result entries."""
    index: dict[str, tuple[str, str]] = {}
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            cid = str(tc.get("id") or fn.get("name") or "")
            index[cid] = (str(fn.get("name") or ""), str(fn.get("arguments") or ""))
    return index


def _last_user_index(messages: list[dict[str, Any]]) -> int | None:
    """Index of the latest user message, or None. Always included in the state,
    regardless of `state.includes` (design decision 2026-11-09)."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "user":
            return i
    return None


def _json_value(content: Any) -> str:
    """Render tool output as a JSON value: parse if valid JSON, else wrap as a JSON string."""
    text = _content_text(content)
    try:
        return json.dumps(json.loads(text), ensure_ascii=False)
    except ValueError:
        return json.dumps(text, ensure_ascii=False)


def _format_state_string(messages: list[dict[str, Any]], includes: dict[str, Any]) -> str:
    """OpenAI-shaped messages -> `role: content` lines. Design doc 4.1.

    The latest user message is always included, regardless of `includes`. Tool
    calls are rendered once, on the tool result entry (name + JSON argument and
    JSON value), so the call is never written twice. The `developer` role counts
    as part of `system` (its settings key) and renders on a `system:` line.
    """
    call_map = _tool_call_index(messages)
    forced = _last_user_index(messages)
    lines: list[str] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        if role == "assistant":
            if not includes.get("assistant"):
                continue
            reasoning = m.get("reasoning_content") or m.get("reasoning")
            if reasoning and includes.get("reasoning"):
                lines.append(f"reasoning: {_escape(_content_text(reasoning))}")
            text = _content_text(m.get("content"))
            if text:
                lines.append(f"assistant: {_escape(text)}")
        elif role == "tool":
            if not includes.get("tool"):
                continue
            name, args = call_map.get(str(m.get("tool_call_id") or ""), ("", ""))
            lines.append(f"tool: {name}({args}) => {_json_value(m.get('content'))}")
        elif role in ("system", "developer", "user"):
            if not includes.get("system" if role != "user" else "user") and i != forced:
                continue
            label = "system" if role != "user" else "user"
            text = _content_text(m.get("content"))
            if text:
                lines.append(f"{label}: {_escape(text)}")
    return "\n".join(lines)


def _format_state_array(messages: list[dict[str, Any]], includes: dict[str, Any]) -> list[dict[str, Any]]:
    """OpenAI-shaped messages -> the entries selected by `includes`, as a list. Design doc 4.2.

    Role labels are required: the array carries them (a `developer` entry keeps
    its role). The latest user message is always included, regardless of
    `includes`; the `developer` role counts as part of `system`.
    """
    out: list[dict[str, Any]] = []
    include_key = {"system": "system", "developer": "system", "user": "user",
                   "assistant": "assistant", "tool": "tool"}
    forced = _last_user_index(messages)
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        if role == "assistant":
            reasoning = m.get("reasoning_content") or m.get("reasoning")
            if reasoning and includes.get("reasoning"):
                out.append({"role": "reasoning", "content": _content_text(reasoning)})
        key = include_key.get(role)
        if key and (includes.get(key) or i == forced):
            entry: dict[str, Any] = {"role": role, "content": m.get("content")}
            if role == "assistant" and m.get("tool_calls"):
                entry["tool_calls"] = m.get("tool_calls")
            if role == "tool":
                entry["tool_call_id"] = m.get("tool_call_id")
            out.append(entry)
    return out


def _format_state(messages: list[dict[str, Any]], settings: dict[str, Any]) -> Any:
    state_cfg = settings.get("state") or {}
    fmt = str(state_cfg.get("format") or "string").strip().lower()
    includes = state_cfg.get("includes")
    includes = includes if isinstance(includes, dict) else {}
    if fmt == "array":
        return _format_state_array(messages, includes)
    return _format_state_string(messages, includes)
