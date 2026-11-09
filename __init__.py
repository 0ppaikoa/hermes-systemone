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

Two registrations, for two resolution paths:

- `custom` — the main loop reaches the plugin through Hermes' custom runtime,
  which passes a URL and no provider name.
- one per `providers:` entry with `api_mode: systemone` — the auxiliary path
  resolves a task's provider by NAME through `PROVIDER_REGISTRY`, so the entry
  has to exist there under its own name.

No core changes. Design notes: systemone-provider-design.md (AokiApp,
2026-11-09). Layout: `state.py` (conversation -> state), `expression.py`
(answers -> assistant message), `client.py` (HTTP), `transport.py` (identity
transport), `profiles.py` (bundled-custom override + URL matching).
"""

from __future__ import annotations

import logging

from agent.transports import register_transport
from providers import register_provider

from .profiles import SystemOneCustomProfile, build_entry_profile, systemone_entries
from .transport import SystemOneTransport

logger = logging.getLogger(__name__)

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


def _register_entry_profiles() -> None:
    """Register a profile per `api_mode: systemone` entry, under the entry's name.

    The auxiliary path (`agent/auxiliary_client.py`) resolves a task's provider
    by name: it looks the name up in `PROVIDER_REGISTRY` and calls that
    profile's `create_client()`. A `providers:` entry is not in that table — it
    is config, not a registered provider — so an auxiliary task naming it falls
    through to the plain OpenAI client and posts to `/chat/completions`, which a
    System One endpoint does not serve.

    Registration must happen at import: `kind: model-provider` plugins are
    imported for their side effects, and `PROVIDER_REGISTRY` is a live view over
    the provider registry, so a name registered here is visible to the resolver.
    A failure is logged and swallowed — discovery must never break startup.
    """
    try:
        entries = systemone_entries()
    except Exception:
        logger.warning("hermes-systemone: could not read providers config at import", exc_info=True)
        return
    for name, entry in entries.items():
        if name == "custom":
            # The bundled name is already registered above and carries the
            # main-loop contract; a second registration would shadow it.
            continue
        try:
            register_provider(build_entry_profile(name, entry))
        except Exception:
            logger.warning("hermes-systemone: could not register provider %r", name, exc_info=True)


_register_entry_profiles()
