"""Tool-side configuration for the ``decide`` tool.

Reads ``plugins.entries.hermes-systemone`` from config.yaml:

    plugins:
      entries:
        hermes-systemone:
          endpoints:
            ollama:
              api: https://ollama.com/v1
              key_env: OLLAMA_API_KEY
          models:
            clef27:
              endpoint: ollama
              base_model: clef:27b
          default_model: clef27

The provider side (``providers:`` entries with ``api_mode: systemone``) is a
separate surface and is not read here. Names are global: every model lives in
one ``models`` map, and the endpoint it belongs to is named by ``endpoint``.
"""

from __future__ import annotations

from typing import Any

PLUGIN_ID = "hermes-systemone"

_ENDPOINT_KEYS = {"api", "key_env"}
_MODEL_KEYS = {"endpoint", "base_model"}


class ToolConfigError(RuntimeError):
    """A configuration mistake the operator has to fix."""


def _as_mapping(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise ToolConfigError(
            f"{path} must be a mapping (got {type(value).__name__})."
        )
    return value


def _as_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolConfigError(f"{path} must be a non-empty string.")
    return value.strip()


def _reject_unknown(mapping: dict, allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ToolConfigError(
            f"{path}: unknown key(s) {unknown}. Allowed: {sorted(allowed)}."
        )


def load_tool_config(cfg: dict | None = None) -> dict:
    """Return the validated tool config.

    Raises :class:`ToolConfigError` with the config path that is wrong. An empty
    ``endpoints`` / ``models`` map is allowed here; a call fails with a clear
    message when it needs an entry that is missing.
    """
    if cfg is None:
        from hermes_cli.config import load_config

        cfg = load_config() or {}

    entries = cfg.get("plugins")
    entries = entries.get("entries") if isinstance(entries, dict) else None
    entry = entries.get(PLUGIN_ID) if isinstance(entries, dict) else None
    if entry is None:
        raise ToolConfigError(
            f"plugins.entries.{PLUGIN_ID} is missing. Add endpoints, models and "
            "default_model to config.yaml."
        )
    entry = _as_mapping(entry, f"plugins.entries.{PLUGIN_ID}")

    raw_endpoints = entry.get("endpoints") or {}
    raw_endpoints = _as_mapping(raw_endpoints, f"plugins.entries.{PLUGIN_ID}.endpoints")
    endpoints: dict[str, dict] = {}
    for name, raw in raw_endpoints.items():
        path = f"plugins.entries.{PLUGIN_ID}.endpoints.{name}"
        raw = _as_mapping(raw, path)
        _reject_unknown(raw, _ENDPOINT_KEYS, path)
        api = _as_text(raw.get("api"), f"{path}.api")
        if not api.startswith(("http://", "https://")):
            raise ToolConfigError(f"{path}.api must start with http:// or https://.")
        endpoints[str(name)] = {"api": api, "key_env": _as_text(raw.get("key_env"), f"{path}.key_env")}

    raw_models = entry.get("models") or {}
    raw_models = _as_mapping(raw_models, f"plugins.entries.{PLUGIN_ID}.models")
    models: dict[str, dict] = {}
    for name, raw in raw_models.items():
        path = f"plugins.entries.{PLUGIN_ID}.models.{name}"
        raw = _as_mapping(raw, path)
        _reject_unknown(raw, _MODEL_KEYS, path)
        endpoint_name = _as_text(raw.get("endpoint"), f"{path}.endpoint")
        if endpoint_name not in endpoints:
            raise ToolConfigError(
                f"{path}.endpoint names {endpoint_name!r}, but no such endpoint exists. "
                f"Configured endpoints: {sorted(endpoints)}."
            )
        models[str(name)] = {
            "endpoint": endpoint_name,
            "base_model": _as_text(raw.get("base_model"), f"{path}.base_model"),
        }

    default_model = entry.get("default_model")
    if default_model is not None:
        default_model = _as_text(default_model, f"plugins.entries.{PLUGIN_ID}.default_model")
        if default_model not in models:
            raise ToolConfigError(
                f"plugins.entries.{PLUGIN_ID}.default_model names {default_model!r}, "
                f"but no such model exists. Configured models: {sorted(models)}."
            )

    return {"endpoints": endpoints, "models": models, "default_model": default_model}


def resolve_model(cfg: dict, requested: Any) -> tuple[str, dict, dict]:
    """Resolve the caller's ``model`` to ``(name, model entry, endpoint entry)``.

    An empty ``model`` falls back to ``default_model``. A name that is not
    configured raises instead of silently using another model.
    """
    name = str(requested or "").strip()
    if not name:
        name = str(cfg.get("default_model") or "").strip()
        if not name:
            raise ToolConfigError(
                "decide: no model was given and default_model is not set. "
                f"Configured models: {sorted(cfg['models'])}."
            )
    model = cfg["models"].get(name)
    if model is None:
        raise ToolConfigError(
            f"decide: model {name!r} is not configured. Configured models: {sorted(cfg['models'])}."
        )
    return name, model, cfg["endpoints"][model["endpoint"]]
