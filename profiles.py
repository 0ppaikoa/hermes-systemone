"""Provider profiles for the System One provider plugin.

Two registrations, because Hermes resolves a provider through different seams
depending on who is asking:

* The bundled ``custom`` profile is overridden so a ``providers:<key>`` entry
  with ``api_mode: systemone`` resolves through Hermes' custom runtime. That is
  the path the **main loop** takes (``_provider_supplied_client``).
* A second profile named after the provider id, carrying ``env_vars``, is
  registered so the **auxiliary resolver** reaches the same ``create_client``.
  ``_resolve_named_custom_branch`` never calls ``create_client``; only
  ``_resolve_api_key_branch`` does, and it is reached through
  ``PROVIDER_REGISTRY``, which mirrors only profiles that declare env vars
  (``auth_plugin_providers._profile_row``). Without the second registration an
  auxiliary route silently falls back to the OpenAI wire and 404s.

Both profiles share ``create_client``: the entry is found by URL, so either
registration serves any ``api_mode: systemone`` entry on that endpoint.

One entry per endpoint. The per-task settings live under ``models.<name>`` and
are resolved per call from the ``model=`` the caller passes, so several tasks may
share one endpoint and one ``base_model``.
"""

from __future__ import annotations

from typing import Any

from .client import SystemOneClient, _models_map, _norm_url

# the bundled module holds an instance; the override class inherits its class.
# `plugins.model_providers.custom` is importable here because plugin discovery
# imports bundled modules before any user plugin (loader order), and this file
# is imported from the plugin module itself.
from plugins.model_providers.custom import custom as _bundled_custom_instance  # noqa: E402

CustomBaseProfile = type(_bundled_custom_instance)


def _entry_url(entry: dict[str, Any]) -> str:
    url_str = _norm_url(entry.get("api") or entry.get("base_url") or entry.get("url") or "")
    return url_str if url_str.endswith("/v1") else url_str + "/v1"


def _entry_mode(entry: dict[str, Any]) -> str:
    return str(entry.get("api_mode") or entry.get("transport") or "").strip().lower()


def _settings_for_base_url(base_url: Any) -> dict[str, Any] | None:
    """Find the `providers:` entry whose URL matches and which declares
    `api_mode: systemone`, else None (no client override). Runs at create_client
    time, when hermes_cli is already loaded; no scan happens at import time.
    URL comparison is normalized: scheme/host/path lowercased, query and
    fragment dropped, so `HTTPS://…` and `…?x=1` still match.

    The first matching entry wins. Two System One entries on one URL are not a
    supported shape: the per-task settings belong under `models.<name>` of a
    single entry, and a second entry on the same URL would be unreachable
    because `create_client` receives no provider name.
    """
    from hermes_cli.config import load_config

    url = _norm_url(base_url)
    if not url:
        return None
    cfg = load_config() or {}
    providers_cfg = cfg.get("providers")
    if isinstance(providers_cfg, dict):
        for entry in providers_cfg.values():
            if not isinstance(entry, dict):
                continue
            if _entry_mode(entry) != "systemone":
                continue
            if _entry_url(entry) == url:
                return entry
    for entry in cfg.get("custom_providers") or []:
        if not isinstance(entry, dict):
            continue
        if _entry_mode(entry) != "systemone":
            continue
        if _entry_url(entry) == url:
            return entry
    return None


def _require_discover_models_off(settings: dict[str, Any], *, base_url: Any) -> None:
    """Refuse to build a client unless `discover_models: false` is set.

    With discovery left on (the default), Hermes prefers the endpoint's live
    `/v1/models` listing over the configured `models` mapping, so the task names
    under `models` disappear from the picker and cannot be selected. The mapping
    is a dict, which `_models_config_is_allowlist()` does not treat as an
    allowlist, so nothing else pins it. Failing here is the only point where the
    mistake is still attributable to this provider.
    """
    discover = settings.get("discover_models")
    if isinstance(discover, str):
        discover = discover.strip().lower() not in {"false", "no", "0"}
    elif isinstance(discover, int) and not isinstance(discover, bool):
        # YAML `0` / `1` parse as ints; `0 is not False` is True, so normalize.
        discover = bool(discover)
    if discover is not False:
        url = _norm_url(base_url) or "?"
        raise RuntimeError(
            "SystemOne provider: providers.<name>.discover_models must be false "
            f"(entry behind {url}). Without it Hermes prefers the endpoint's live "
            "/v1/models listing and the task names under models are not selectable. "
            "Add `discover_models: false` to the entry."
        )


class SystemOneCustomProfile(CustomBaseProfile):
    """Override of the bundled ``custom`` profile: serves the SystemOne client for
    ``api_mode: systemone`` routes.

    The System One API has no streaming form. Declaring ``supports_streaming =
    False`` is Hermes' standard opt-out (``turn_api_call._should_stream`` reads
    the profile attribute); without it the main loop picks the streaming path
    and the first attempt of every turn fails before falling back.
    """

    supports_streaming = False

    def create_client(self, **client_kwargs: Any) -> Any:
        settings = _settings_for_base_url(client_kwargs.get("base_url"))
        if settings is None:
            return None
        _require_discover_models_off(settings, base_url=client_kwargs.get("base_url"))
        _models_map(settings)
        allowed = {"api_key", "base_url", "timeout", "http_client", "default_headers"}
        return SystemOneClient(
            settings=settings,
            **{key: value for key, value in client_kwargs.items() if key in allowed},
        )


def make_named_profile(provider_id: str, *, key_env: str, base_url: str) -> SystemOneCustomProfile:
    """A profile registered under *provider_id*, so the auxiliary resolver finds it.

    ``PROVIDER_REGISTRY`` mirrors a plugin profile only when it declares
    ``env_vars`` (``auth_plugin_providers._profile_row`` returns None otherwise),
    and only a mirrored row reaches ``_resolve_api_key_branch`` — the one
    auxiliary branch that calls ``create_client``. The bundled ``custom``
    profile carries no env vars and is resolved by
    ``_resolve_named_custom_branch``, which never calls ``create_client``, so the
    auxiliary route needs this second registration.

    ``create_client`` is inherited unchanged: it locates the entry by URL, so the
    named profile serves the same ``api_mode: systemone`` endpoints the ``custom``
    override does.
    """
    profile = SystemOneCustomProfile(
        name=provider_id,
        aliases=(),
        env_vars=(key_env,),
        base_url=base_url,
        auth_type="api_key",
    )
    # dataclass field default (None) wins over the class attribute, so set it on
    # the instance — the main loop reads this to skip the streaming attempt.
    profile.supports_streaming = False
    return profile
