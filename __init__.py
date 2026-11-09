"""System One provider plugin for Hermes Agent.

System One is a decision-only API served at `POST /v1/systemone`. Several
vendors and self-hosted servers speak it, and it never generates text, so this
plugin:

- sends the OpenAI-shaped conversation history as the `state` (shape from
  `providers.<name>.state`), see `state.py`,
- assembles questions from `providers.<name>.questions`; missing settings are
  errors, not silent defaults,
- re-assembles the returned `answers` into an assistant message through the
  projection expression in `providers.<name>.response.expression`, see
  `expression.py`.

No core changes. Design notes: systemone-provider-design.md (AokiApp,
2026-11-09). Layout: `state.py` (conversation -> state), `expression.py`
(answers -> assistant message), `client.py` (HTTP), `transport.py` (identity
transport), `profiles.py` (bundled-custom override + URL matching).
"""

from __future__ import annotations

from agent.transports import register_transport
from providers import register_provider

from .profiles import SystemOneCustomProfile
from .transport import SystemOneTransport

__all__ = ["SystemOneTransport", "SystemOneCustomProfile"]

register_transport("systemone", SystemOneTransport)
register_provider(SystemOneCustomProfile(
    name="custom",
    aliases=(),  # local-server alias names already map to custom in the core alias table
    env_vars=(),
    base_url="",
))
