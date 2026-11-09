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


def _is_systemone(entry: dict[str, Any]) -> bool:
    mode = str(entry.get("api_mode") or entry.get("transport") or "").strip().lower()
    return mode == "systemone"


def _settings_for_base_url(base_url: Any) -> list[dict[str, Any]]:
    """Every `providers:` entry whose URL matches and which declares
    `api_mode: systemone`, in config order. Empty means no client override.

    A list, not one entry: two System One providers may legitimately share one
    endpoint (a classifier and a guard, say), and only the model tells them
    apart. The client picks per call; the URL alone never decides.

    Runs at create_client time, when hermes_cli is already loaded; no scan
    happens at import time. URL comparison is normalized: scheme/host/path
    lowercased, query and fragment dropped, so `HTTPS://…` and `…?x=1` match.
    """
    from hermes_cli.config import load_config

    url = _norm_url(base_url)
    if not url:
        return []
    cfg = load_config() or {}
    found: list[dict[str, Any]] = []
    providers_cfg = cfg.get("providers")
    if isinstance(providers_cfg, dict):
        for key, entry in providers_cfg.items():
            if not isinstance(entry, dict) or not _is_systemone(entry):
                continue
            if _entry_url(entry) == url:
                found.append({"_key": str(key), **entry})
    for entry in cfg.get("custom_providers") or []:
        if not isinstance(entry, dict) or not _is_systemone(entry):
            continue
        if _entry_url(entry) == url:
            key = str(entry.get("provider_key") or entry.get("name") or "")
            found.append({"_key": key, **entry})
    return found


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
        candidates = _settings_for_base_url(client_kwargs.get("base_url"))
        if not candidates:
            return None
        allowed = {"api_key", "base_url", "timeout", "http_client", "default_headers"}
        return SystemOneClient(
            candidates=candidates,
            **{key: value for key, value in client_kwargs.items() if key in allowed},
        )
