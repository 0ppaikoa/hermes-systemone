"""HTTP client for the System One provider plugin.

An OpenAI-SDK-shaped facade over `POST /v1/systemone`: state assembly, question
build, and answer-to-message conversion happen here. Also hosts the URL
normalization helpers shared with `profiles.py`.

Config shape (one provider, one endpoint, several named tasks):

    providers:
      systemone:
        api: https://ollama.com/v1
        api_mode: systemone
        discover_models: false
        default_model: imi
        models:
          imi:
            base_model: clef:27b
            state: {...}
            questions: {...}
            response: {expression: "..."}

`models` keys are the names callers pass as `model=`; `base_model` is the id
sent to the endpoint. Two tasks may share one `base_model`.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
from .expression import _assemble, _eval_expression
from .state import _format_state

API_PATH = "/systemone"


def _models_map(settings: dict[str, Any]) -> dict[str, Any]:
    """The `models` mapping of a provider entry, or a RuntimeError explaining what to write."""
    models = settings.get("models")
    if not isinstance(models, dict) or not models:
        raise RuntimeError(
            "SystemOne provider: no models configured. "
            "Write providers.<name>.models.<task-name> with base_model, questions "
            "and response.expression in config.yaml."
        )
    return models


def _resolve_task(settings: dict[str, Any], model: Any) -> tuple[str, dict[str, Any]]:
    """Map the caller's `model=` to `(task_name, task_config)`.

    An empty `model` means the caller expressed no preference, so `default_model`
    stands in. Any other value must name a key under `models`; a name that is not
    there is a config mistake and raises rather than answering with another
    task's questions.
    """
    models = _models_map(settings)
    name = str(model or "").strip()
    if not name:
        name = str(settings.get("default_model") or "").strip()
        if not name:
            raise RuntimeError(
                "SystemOne provider: the caller passed no model and the entry sets no "
                "default_model. Set providers.<name>.default_model to one of "
                f"{sorted(models)}."
            )
    config = models.get(name)
    if config is None:
        raise RuntimeError(
            f"SystemOne provider: no task named {name!r} under providers.<name>.models. "
            f"Configured: {sorted(models)}."
        )
    if not isinstance(config, dict):
        raise RuntimeError(
            f"SystemOne provider: providers.<name>.models.{name} must be a mapping with "
            "base_model, questions and response.expression."
        )
    return name, config


def _base_model(task_name: str, config: dict[str, Any]) -> str:
    """The model id sent to the endpoint. Required per task; no inherited default."""
    base = config.get("base_model")
    if not isinstance(base, str) or not base.strip():
        raise RuntimeError(
            f"SystemOne provider: providers.<name>.models.{task_name}.base_model is required. "
            "Set it to the model id the endpoint serves (for example clef:27b)."
        )
    return base.strip()


def _require_settings(settings: dict[str, Any], *, part: str, task: str = "") -> None:
    where = f"providers.<name>.models.{task}." if task else "providers.<name>."
    if part == "questions":
        questions = settings.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise RuntimeError(
                f"SystemOne provider: no questions configured. Write {where}questions in config.yaml."
            )
        return
    expression = (settings.get("response") or {}).get("expression")
    if not (isinstance(expression, str) and expression.strip()):
        raise RuntimeError(
            f"SystemOne provider: no projection expression configured. "
            f"Write {where}response.expression in config.yaml."
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
    task resolution, state assembly, question build, and answer-to-message
    conversion happen inside this client. Auxiliary tasks (compression, titles)
    run through the same client. Errors are plain RuntimeError (not Hermes
    ProviderError subclasses); the auxiliary resolver catches them, and the main
    loop surfaces them as-is.
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

        task, config = _resolve_task(self._settings, model)
        base_model = _base_model(task, config)
        _require_settings(config, part="questions", task=task)
        _require_settings(config, part="expression", task=task)
        questions = config.get("questions") or {}
        state = _format_state(list(messages or []), config)
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
            url,
            json={"model": base_model, "state": state, "questions": questions},
            **post_kwargs,
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
        expression = (config.get("response") or {}).get("expression")
        answers = data.get("answers") or {}
        result = _eval_expression(str(expression).strip(), answers)
        return _assemble(result, answers, data.get("usage") or {}, base_model)
