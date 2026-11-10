"""System One provider plugin for Hermes Agent.

System One is a decision-only API served at `POST /v1/systemone`. Several
vendors and self-hosted servers speak it, and it never generates text, so this
plugin:

- sends the OpenAI-shaped conversation history as the `state` (shape from
  `providers.<name>.models.<task>.state`), see `state.py`,
- assembles questions from `models.<task>.questions`; missing settings are
  errors, not silent defaults,
- re-assembles the returned `answers` into an assistant message through the
  projection expression in `models.<task>.response.expression`, see
  `expression.py`.

One provider entry serves one endpoint. Each task lives under `models.<name>`,
where the key is the name callers pass as `model=` and `base_model` is the id
the endpoint serves; several tasks may share one `base_model`.

No core changes. Design notes: systemone-provider-design.md (AokiApp,
2026-11-09). Layout: `state.py` (conversation -> state), `expression.py`
(answers -> assistant message), `client.py` (HTTP), `transport.py` (identity
transport), `profiles.py` (bundled-custom override + URL matching),
`aux_route.py` (auxiliary-route wiring).

`aux_route.py` closes a gap between the main loop and the auxiliary resolver:
both build a client for a named custom provider, but only the main loop asks
`registered_api_modes()` before deciding the wire. Without the patch an
`api_mode: systemone` auxiliary route posts to `/v1/chat/completions`.
"""

from __future__ import annotations

from agent.transports import register_transport
from providers import register_provider

from . import aux_route  # noqa: F401  (installs the auxiliary-route wiring)
from .profiles import SystemOneCustomProfile
from .transport import SystemOneTransport

__all__ = ["SystemOneTransport", "SystemOneCustomProfile"]

register_transport("systemone", SystemOneTransport)
_SYSTEMONE_PROFILE = SystemOneCustomProfile(
    name="custom",
    aliases=(),  # local-server alias names already map to custom in the core alias table
    env_vars=(),
    base_url="",
)
# dataclass field default (None) wins over the class attribute, and a
# subclass __post_init__ is never invoked without re-decoration, so set it
# on the instance: the main loop reads this to skip the streaming attempt.
_SYSTEMONE_PROFILE.supports_streaming = False
register_provider(_SYSTEMONE_PROFILE)
