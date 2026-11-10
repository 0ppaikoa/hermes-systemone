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
transport), `profiles.py` (profile registrations + URL matching).

## Two registrations, one `create_client`

Hermes resolves a provider through different seams depending on the caller, and
only one of them consults a plugin profile's `create_client`:

* **Main loop** — `agent_runtime_helpers._provider_supplied_client` calls
  `profile.create_client` for the resolved provider. A `providers:<key>` entry
  with `api_mode: systemone` resolves through the bundled `custom` profile, so
  overriding `custom` covers this path.
* **Auxiliary resolver** — `_resolve_named_custom_branch` builds a plain OpenAI
  client and never calls `create_client`. Only `_resolve_api_key_branch` does,
  and it is reached through `PROVIDER_REGISTRY`, which mirrors a plugin profile
  only when it declares `env_vars`.

So the plugin registers `custom` (main loop) *and* one profile per configured
`api_mode: systemone` entry, named after that entry and carrying its `key_env`
(auxiliary). Both share `create_client`, which locates the entry by URL.
"""

from __future__ import annotations

from typing import Any

from agent.transports import register_transport
from providers import register_provider

from .profiles import SystemOneCustomProfile, make_named_profile
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


def _systemone_entries() -> list[tuple[str, str, str]]:
    """``(provider_id, key_env, base_url)`` for every `api_mode: systemone` entry.

    Read from config at import time. A config that cannot be read yields no
    named profiles — the `custom` override still serves the main loop, and the
    auxiliary route keeps whatever behaviour it had.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
    except Exception:
        return []
    entries: list[tuple[str, str, str]] = []
    providers_cfg = cfg.get("providers")
    if isinstance(providers_cfg, dict):
        for name, entry in providers_cfg.items():
            if not isinstance(entry, dict):
                continue
            mode = str(entry.get("api_mode") or entry.get("transport") or "").strip().lower()
            if mode != "systemone":
                continue
            key_env = str(entry.get("key_env") or "").strip()
            base_url = str(entry.get("base_url") or entry.get("api") or "").strip()
            if not key_env or not base_url:
                # Without a key env the registry mirror skips the row, so a named
                # profile would do nothing; without a base URL the entry cannot be
                # located by `_settings_for_base_url`.
                continue
            entries.append((str(name), key_env, base_url))
    return entries


for _provider_id, _key_env, _base_url in _systemone_entries():
    try:
        register_provider(make_named_profile(_provider_id, key_env=_key_env, base_url=_base_url))
    except Exception:
        # A duplicate or malformed id must not stop the plugin from loading: the
        # `custom` override is already registered and serves the main loop.
        import logging

        logging.getLogger(__name__).warning(
            "systemone: could not register the auxiliary profile for %r", _provider_id,
            exc_info=True,
        )
