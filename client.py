"""HTTP client for the System One provider plugin.

An OpenAI-SDK-shaped facade over `POST /v1/systemone`: state assembly, question
build, and answer-to-message conversion happen here. Also hosts the URL
normalization helpers shared with `profiles.py`.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import httpx
from .expression import _assemble, _eval_expression
from .state import _format_state

API_PATH = "/systemone"


def _require_settings(settings: dict[str, Any], *, part: str) -> None:
    questions = settings.get("questions")
    if part == "questions":
        if not isinstance(questions, dict) or not questions:
            raise RuntimeError(
                "SystemOne provider: no questions configured. "
                "Write providers.<name>.questions in config.yaml."
            )
        return
    expression = (settings.get("response") or {}).get("expression")
    if not (isinstance(expression, str) and expression.strip()):
        raise RuntimeError(
            "SystemOne provider: no projection expression configured. "
            "Write providers.<name>.response.expression in config.yaml."
        )


def _norm_url(raw: Any) -> str:
    """Normalize an endpoint URL: lowercase scheme, host, and path; drop query and fragment."""
    from urllib.parse import urlsplit

    url_str = str(raw or "").strip()
    if not url_str:
        return ""
    parts = urlsplit(url_str)
    rebuilt = f"{parts.scheme.lower()}://{parts.netloc.lower()}{parts.path.lower()}"
    return rebuilt.rstrip("/")


def _v1_base(base: Any) -> str:
    """`_norm_url` plus the `/v1` tail: `https://example.com` -> `https://example.com/v1`."""
    base = _norm_url(base)
    return base if base.endswith("/v1") else base + "/v1"


def _entry_models(entry: dict[str, Any]) -> set[str]:
    """Every model id this entry claims: `default_model`, `model`, and the `models` catalog.

    Lowercased, because config spelling and the wire id drift in case. An entry
    that claims no model at all returns an empty set and can never win a
    multi-candidate pick — that is deliberate: an unnamed entry on a shared
    endpoint is ambiguous, and guessing is worse than an error.
    """
    out: set[str] = set()
    for key in ("default_model", "model"):
        value = str(entry.get(key) or "").strip()
        if value:
            out.add(value.lower())
    models = entry.get("models")
    if isinstance(models, dict):
        out.update(str(name).strip().lower() for name in models if str(name).strip())
    elif isinstance(models, (list, tuple)):
        out.update(str(name).strip().lower() for name in models if str(name).strip())
    return out


class SystemOneClient:
    """OpenAI-SDK-shaped client over the System One API (POST /v1/systemone).

    Hermes sees the usual `client.chat.completions.create(**kwargs)` surface;
    state assembly, question build, and answer-to-message conversion happen
    inside this client. Auxiliary tasks (compression, titles) run through the
    same client. Errors are plain RuntimeError (not Hermes ProviderError
    subclasses); the auxiliary resolver catches them, and the main loop
    surfaces them as-is.

    Several `api_mode: systemone` entries may share one base_url. The client
    then holds every candidate on that URL and picks one per call from the
    `model` the caller passes, because the URL alone cannot tell them apart.
    """

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, *, api_key: str = "", base_url: Any = None, settings: dict[str, Any] | None = None,
                 candidates: list[dict[str, Any]] | None = None,
                 default_headers: Any = None, timeout: Any = None, http_client: Any = None, **_: Any) -> None:
        self.api_key = api_key
        self.base_url = _norm_url(base_url)
        if candidates is not None:
            self._candidates = [entry for entry in candidates if isinstance(entry, dict)]
        elif settings is not None:
            self._candidates = [settings]
        else:
            self._candidates = []
        self._settings = self._candidates[0] if len(self._candidates) == 1 else {}
        self._default_headers = dict(default_headers or {})
        self.is_closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._http = http_client or httpx.Client(timeout=timeout or 120.0)

    def _settings_for_model(self, model: str) -> dict[str, Any]:
        """The entry this call belongs to.

        One candidate on the URL: that one, whatever the model says — the
        single-entry setup keeps working exactly as before. Several: the entry
        that claims `model`. Ambiguity is an error, never a silent pick — a
        wrong pick means sending another provider's questions.
        """
        if not self._candidates:
            raise RuntimeError(
                "SystemOne provider: no `api_mode: systemone` entry matches this base_url."
            )
        if len(self._candidates) == 1:
            return self._candidates[0]
        wanted = str(model or "").strip().lower()
        names = [str(entry.get("_key") or entry.get("name") or "?") for entry in self._candidates]
        if not wanted:
            raise RuntimeError(
                "SystemOne provider: several System One providers share this base_url "
                f"({names}) and no model was given to choose one."
            )
        hits = [entry for entry in self._candidates if wanted in _entry_models(entry)]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise RuntimeError(
                "SystemOne provider: several System One providers share this base_url "
                f"({names}) and none claims model {model!r}. Set `default_model` "
                "(or list it under `models`) on the intended entry."
            )
        raise RuntimeError(
            f"SystemOne provider: model {model!r} is claimed by several entries on this "
            f"base_url ({[str(e.get('_key') or e.get('name') or '?') for e in hits]}). "
            "Give each entry a distinct model."
        )

    def _url(self) -> str:
        base = self.base_url
        if not base:
            raise RuntimeError(
                "SystemOne provider: base_url is not configured. "
                "Set providers.<name>.base_url to the API endpoint."
            )
        return _v1_base(base) + API_PATH

    def _headers(self) -> dict[str, str]:
        """Merged request headers.

        ``self._default_headers`` arrives from Hermes' client construction and
        may carry ``openai.Omit`` values — the SDK's "send nothing" sentinel.
        The SDK drops such values itself when building a request; plain httpx
        does not and raises ``TypeError`` on them, so anything that is not a
        ``str`` is treated as absent here.
        """
        merged = {
            key: value
            for key, value in self._default_headers.items()
            if isinstance(value, str)
        }
        headers = {"Content-Type": "application/json", **merged}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def close(self) -> None:
        self.is_closed = True
        try:
            self._http.close()
        except Exception:
            pass

    _require_settings = staticmethod(_require_settings)

    def _create(self, *, model: str = "", messages: Any = None, stream: bool = False, **kwargs: Any) -> Any:
        if stream:
            raise RuntimeError("SystemOne provider: streaming is not supported (supports_streaming=False).")

        settings = self._settings_for_model(model)
        _require_settings(settings, part="questions")
        _require_settings(settings, part="expression")
        questions = settings.get("questions") or {}
        state = _format_state(list(messages or []), settings)
        url = self._url()
        response = self._http.post(url, json={"model": model, "state": state, "questions": questions},
                                   headers=self._headers())
        if response.status_code != 200:
            raise RuntimeError(
                f"SystemOne API error: HTTP {response.status_code}: {response.text[:300]}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"SystemOne API error: HTTP {response.status_code} returned a non-JSON body: "
                f"{str(response.text or '')[:200]}"
            ) from exc
        expression = (settings.get("response") or {}).get("expression")
        answers = data.get("answers") or {}
        result = _eval_expression(str(expression).strip(), answers)
        return _assemble(result, answers, data.get("usage") or {}, model)
