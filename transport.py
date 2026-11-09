"""Identity transport for the System One provider plugin.

Message conversion happens in the client (`client.py`), so this layer passes
the OpenAI-shaped structures through unchanged.
"""

from __future__ import annotations

from typing import Any

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall, Usage


class SystemOneTransport(ProviderTransport):
    """`api_mode='systemone'` transport: identity conversion, client-facade dispatch."""

    @property
    def api_mode(self) -> str:
        return "systemone"

    def convert_messages(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return messages

    def convert_tools(self, tools: list[dict[str, Any]] | None) -> Any:
        return tools or []

    def build_kwargs(self, model: str, messages: list[dict[str, Any]], tools: Any = None, **params: Any) -> dict:
        return {"model": model, "messages": messages}

    def normalize_response(self, response: Any, **kwargs: Any) -> NormalizedResponse:
        choice = response.choices[0]
        msg = choice.message
        calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ] or None
        return NormalizedResponse(
            content=msg.content,
            tool_calls=calls,
            finish_reason=choice.finish_reason,
            reasoning=getattr(msg, "reasoning_content", None),
            usage=Usage.from_openai(response.usage) if getattr(response, "usage", None) else None,
        )
