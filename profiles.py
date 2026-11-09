"""Provider profiles for the System One provider plugin.

The plugin overrides the bundled "custom" profile: a `providers:<key>` entry
with `api_mode: systemone` resolves through Hermes' custom runtime (whose
provider name is "custom"), and `create_client` serves the SystemOne client
only when the entry behind the URL declares `api_mode: systemone`.

One entry per endpoint. The per-task settings live under `models.<name>` and
are resolved per call from the `model=` the caller passes, so several tasks may
share one endpoint and one `base_model`.
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


def _settings_for_provider_name(name: Any) -> dict[str, Any] | None:
    """The `providers:<name>` entry, when it declares `api_mode: systemone`.

    The auxiliary path resolves a task's provider by NAME, not by URL: a
    registered profile is looked up in `PROVIDER_REGISTRY` and its
    `create_client()` is called with the entry's own `api_key` / `base_url`.
    Matching on the name keeps the two paths pointing at the same entry even
    when several `providers:` entries share one endpoint.
    """
    from hermes_cli.config import load_config

    key = str(name or "").strip()
    if not key:
        return None
    entry = ((load_config() or {}).get("providers") or {}).get(key)
    if not isinstance(entry, dict) or _entry_mode(entry) != "systemone":
        return None
    return entry


def _entry_key_env(entry: dict[str, Any]) -> str:
    return str(entry.get("key_env") or entry.get("api_key_env") or "").strip()


def systemone_entries() -> dict[str, dict[str, Any]]:
    """Every `providers:` entry that declares `api_mode: systemone`, keyed by name."""
    from hermes_cli.config import load_config

    cfg = load_config() or {}
    entries = cfg.get("providers")
    if not isinstance(entries, dict):
        return {}
    return {
        str(key): entry
        for key, entry in entries.items()
        if isinstance(entry, dict) and _entry_mode(entry) == "systemone"
    }


def build_entry_profile(name: str, entry: dict[str, Any]) -> "SystemOneCustomProfile":
    """A profile registered under the entry's own name, for the auxiliary path.

    `env_vars` must not be empty: `PROVIDER_REGISTRY` — the table the auxiliary
    resolver reads `auth_type` from — only holds providers that declare at least
    one credential variable, so a profile with `env_vars=()` is invisible there
    and the task falls through to the plain OpenAI client. The entry's `key_env`
    is the honest value; the fallback name is only for entries that rely on an
    inline `api_key`.
    """
    key_env = _entry_key_env(entry)
    profile = SystemOneCustomProfile(
        name=name,
        aliases=(),
        env_vars=(key_env,) if key_env else ("SYSTEMONE_API_KEY",),
        base_url=str(entry.get("api") or entry.get("base_url") or entry.get("url") or ""),
        auth_type="api_key",
    )
    profile.supports_streaming = False
    return profile


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
        # The auxiliary path calls this on a profile registered under the entry's
        # own name, and passes that entry's api_key / base_url; the main loop
        # reaches this same class through the "custom" profile with only a URL.
        # Try the name first so both paths land on the same entry.
        settings = _settings_for_provider_name(self.name)
        if settings is None:
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
