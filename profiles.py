"""Provider profiles for the System One provider plugin.

The plugin overrides the bundled "custom" profile: a `providers:<key>` entry
with `api_mode: systemone` resolves through Hermes' custom runtime (whose
provider name is "custom"), and `create_client` serves the SystemOne client
only when the entry behind the URL declares `api_mode: systemone`.
"""

from __future__ import annotations

from typing import Any

from .client import SystemOneClient, _norm_url, _v1_base

# the bundled module holds an instance; the override class inherits its class.
# `plugins.model_providers.custom` is importable here because plugin discovery
# imports bundled modules before any user plugin (loader order), and this file
# is imported from the plugin module itself.
from plugins.model_providers.custom import custom as _bundled_custom_instance  # noqa: E402

CustomBaseProfile = type(_bundled_custom_instance)


def _entry_url(entry: dict[str, Any]) -> str:
    url_str = _norm_url(entry.get("api") or entry.get("base_url") or entry.get("url") or "")
    return url_str if url_str.endswith("/v1") else url_str + "/v1"


def _settings_for_base_url(base_url: Any) -> dict[str, Any] | None:
    """Find the `providers:` entry whose URL matches and which declares
    `api_mode: systemone`, else None (no client override). Runs at create_client
    time, when hermes_cli is already loaded; no scan happens at import time.
    URL comparison is normalized: scheme/host/path lowercased, query and
    fragment dropped, so `HTTPS://…` and `…?x=1` still match.
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
            mode = str(entry.get("api_mode") or entry.get("transport") or "").strip().lower()
            if mode != "systemone":
                continue
            if _entry_url(entry) == url:
                return entry
    for entry in cfg.get("custom_providers") or []:
        if not isinstance(entry, dict):
            continue
        mode = str(entry.get("api_mode") or entry.get("transport") or "").strip().lower()
        if mode != "systemone":
            continue
        if _entry_url(entry) == url:
            return entry
    return None


class SystemOneCustomProfile(CustomBaseProfile):
    """Override of the bundled ``custom`` profile: serves the SystemOne client for
    ``api_mode: systemone`` routes.

    The System One API has no streaming form. Declaring ``supports_streaming =
    False`` is Hermes' standard opt-out (``turn_api_call._should_stream`` reads
    the profile attribute); without it the main loop picks the streaming path
    and the first attempt of every turn fails before falling back.
    """

    supports_streaming = False

    def __post_init__(self) -> None:
        # The parent is a dataclass: a bare class attribute is shadowed by the
        # instance field (default None), so set it on the instance here.
        self.supports_streaming = False

    def create_client(self, **client_kwargs: Any) -> Any:
        settings = _settings_for_base_url(client_kwargs.get("base_url"))
        if settings is None:
            return None
        allowed = {"api_key", "base_url", "timeout", "http_client", "default_headers"}
        return SystemOneClient(
            settings=settings,
            **{key: value for key, value in client_kwargs.items() if key in allowed},
        )
