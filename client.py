"""HTTP client for the System One provider plugin.

An OpenAI-SDK-shaped facade over `POST /v1/systemone`: state assembly, question
build, and answer-to-message conversion happen here. Also hosts the URL
normalization helpers shared with `profiles.py`.
"""

from __future__ import annotations

import json
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


class SystemOneClient:
    """OpenAI-SDK-shaped client over the System One API (POST /v1/systemone).

    Hermes sees the usual `client.chat.completions.create(**kwargs)` surface;
    state assembly, question build, and answer-to-message conversion happen
    inside this client. Auxiliary tasks (compression, titles) run through the
    same client. Errors are plain RuntimeError (not Hermes ProviderError
    subclasses); the auxiliary resolver catches them, and the main loop
    surfaces them as-is.
    """

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, *, api_key: str = "", base_url: Any = None, settings: dict[str, Any] | None = None,
                 default_headers: Any = None, timeout: Any = None, http_client: Any = None, **_: Any) -> None:
        self.api_key = api_key
        self.base_url = _norm_url(base_url)
        self._settings = settings or {}
        self._default_headers = dict(default_headers or {})
        self.is_closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._http = http_client or httpx.Client(timeout=timeout or 120.0)

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

        settings = self._settings
        _require_settings(settings, part="questions")
        _require_settings(settings, part="expression")
        questions = settings.get("questions") or {}
        state = _format_state(list(messages or []), settings)
        url = self._url()
        # Per-request timeout. Hermes passes one on every call (an explicit
        # `timeout=` kwarg from the auxiliary layer, e.g. auxiliary.<task>.timeout);
        # httpx applies it to this request only, leaving the client's own default
        # alone. Dropping it silently pinned every call to the construction-time
        # default, so a stalled endpoint held the caller for the full 120s.
        post_kwargs: dict[str, Any] = {"headers": self._headers()}
        if kwargs.get("timeout") is not None:
            post_kwargs["timeout"] = kwargs["timeout"]
        response = self._http.post(
            url, json={"model": model, "state": state, "questions": questions}, **post_kwargs
        )
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
