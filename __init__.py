"""System One (System 1) provider plugin for Hermes Agent.

System One is a decision-only API served at `POST /v1/systemone`. Several
vendors and self-hosted servers speak it, and it never generates text, so this
plugin:

- sends the OpenAI-shaped conversation history as the `state` (shape from
  `providers.<name>.state`),
- assembles questions from `providers.<name>.questions`; missing settings are
  errors, not silent defaults,
- re-assembles the returned `answers` into an assistant message through the
  projection expression in `providers.<name>.response.expression`.

No core changes. Design notes: systemone-provider-design.md (AokiApp,
2026-11-09).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from agent.transports import register_transport
from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall, Usage
from providers import register_provider
from providers.base import ProviderProfile
from plugins.model_providers.custom import custom as _bundled_custom_instance

# the bundled module holds an instance; the override class inherits its class
CustomBaseProfile = type(_bundled_custom_instance)

API_PATH = "/systemone"


def _v1_base(base: Any) -> str:
    """Normalize base_url to end with /v1: `https://example.com` -> `https://example.com/v1`."""
    base = str(base or "").strip().rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"

# --- settings ----------------------------------------------------------------


def _settings_for(provider: str) -> dict[str, Any]:
    """Read `providers.<name>` (or a legacy `custom_providers:` entry) by provider id.

    Returns {} when absent; missing questions / expression raise at request time.
    """
    from hermes_cli.config import load_config

    target = str(provider or "").strip().lower()
    if not target:
        return {}
    cfg = load_config() or {}
    providers = cfg.get("providers")
    if isinstance(providers, dict):
        for key, value in providers.items():
            if str(key).strip().lower() == target and isinstance(value, dict):
                return value
    for entry in cfg.get("custom_providers") or []:
        if isinstance(entry, dict) and str(entry.get("name") or "").strip().lower() == target:
            return entry
    return {}


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


# --- state -------------------------------------------------------------------

_ROLES = ("system", "user", "assistant", "tool", "reasoning")


def _escape(text: Any) -> str:
    """JSON-escape content so it stays one line; strips the outer quotes."""
    return json.dumps(str(text if text is not None else ""), ensure_ascii=False)[1:-1]


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            elif part is not None:
                parts.append(str(part))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _tool_call_index(messages: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """tool_call_id -> (name, arguments), for building tool result entries."""
    index: dict[str, tuple[str, str]] = {}
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            cid = str(tc.get("id") or fn.get("name") or "")
            index[cid] = (str(fn.get("name") or ""), str(fn.get("arguments") or ""))
    return index


def _format_state_string(messages: list[dict[str, Any]], includes: dict[str, Any]) -> str:
    """OpenAI-shaped messages -> `role: content` lines. Design doc section 4.1."""
    call_map = _tool_call_index(messages)
    lines: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        if role == "assistant":
            reasoning = m.get("reasoning_content") or m.get("reasoning")
            if reasoning and includes.get("reasoning"):
                lines.append(f"reasoning: {_escape(_content_text(reasoning))}")
            text = _content_text(m.get("content"))
            if text:
                lines.append(f"assistant: {_escape(text)}")
            for tc in m.get("tool_calls") or []:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                call = f"{fn.get('name')}({fn.get('arguments')})"
                lines.append(f"tool: {_escape(call)}")
        elif role == "tool":
            name, args = call_map.get(str(m.get("tool_call_id") or ""), ("", ""))
            result = _escape(_content_text(m.get("content")))
            lines.append(f"tool: {_escape(f'{name}({args})')} => {result}")
        elif role in ("system", "user"):
            if not includes.get(role):
                continue
            text = _content_text(m.get("content"))
            if text:
                lines.append(f"{role}: {_escape(text)}")
    return "\n".join(lines)


def _format_state_array(messages: list[dict[str, Any]], includes: dict[str, Any]) -> list[dict[str, Any]]:
    """OpenAI-shaped messages -> the entries selected by `includes`, as a list. Design doc 4.2.

    Role labels are required: the array carries them.
    """
    out: list[dict[str, Any]] = []
    role_of = {"system": "system", "user": "user", "assistant": "assistant", "tool": "tool"}
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        if role == "assistant":
            reasoning = m.get("reasoning_content") or m.get("reasoning")
            if reasoning and includes.get("reasoning"):
                out.append({"role": "reasoning", "content": _content_text(reasoning)})
        if role in role_of and includes.get(role):
            entry: dict[str, Any] = {"role": role, "content": m.get("content")}
            if role == "assistant" and m.get("tool_calls"):
                entry["tool_calls"] = m.get("tool_calls")
            if role == "tool":
                entry["tool_call_id"] = m.get("tool_call_id")
            out.append(entry)
    return out


def _format_state(messages: list[dict[str, Any]], settings: dict[str, Any]) -> Any:
    state_cfg = settings.get("state") or {}
    fmt = str(state_cfg.get("format") or "string").strip().lower()
    includes = state_cfg.get("includes")
    includes = includes if isinstance(includes, dict) else {}
    if fmt == "array":
        return _format_state_array(messages, includes)
    return _format_state_string(messages, includes)


# --- response assembly -------------------------------------------------------

def _eval_expression(expression: str, answers: dict[str, Any]) -> Any:
    """Evaluate the projection expression with simpleeval. Namespaces: `answers`
    (plain data) plus a whitelist of builtins. Design doc 5.1.

    simpleeval's default evaluator does not evaluate dict / list literals
    (FeatureNotAvailable), and the projection expression returns a dict
    (design doc 5.2), so exactly those nodes are added. Comprehensions stay
    rejected.
    """
    import ast as _ast

    import simpleeval

    functions = {
        "len": len,
        "round": round,
        "int": int,
        "float": float,
        "str": str,
        # String methods ('%.2f'.format(x)) are rejected by simpleeval, so the
        # two-argument builtin format(value, spec) is whitelisted instead.
        "format": format,
        "json_dumps": lambda value: json.dumps(value, ensure_ascii=False),
    }
    evaluator = simpleeval.SimpleEval(
        names={"answers": answers},
        functions=functions,
    )
    evaluator.nodes.update({
        _ast.Dict: lambda node: {
            evaluator._eval(key): evaluator._eval(value)
            for key, value in zip(node.keys, node.values) if key is not None
        },
        _ast.List: lambda node: [evaluator._eval(elt) for elt in node.elts],
        _ast.Tuple: lambda node: tuple(evaluator._eval(elt) for elt in node.elts),
    })
    return evaluator.eval(expression)


_MESSAGE_KEYS = ("content", "tool_calls", "reasoning", "finish_reason")


def _assemble(expression_result: Any, answers: dict[str, Any], usage: dict[str, Any], model: str) -> SimpleNamespace:
    """Projection result (dict or str) -> OpenAI-shaped response. Design doc 5.2 / 5.3."""
    content: Any = None
    tool_calls: list[Any] = []
    reasoning: Any = None
    finish_reason = "stop"
    if isinstance(expression_result, str):
        content = expression_result
    elif isinstance(expression_result, dict):
        unknown = [key for key in expression_result if key not in _MESSAGE_KEYS]
        if unknown:
            raise RuntimeError(
                "SystemOne provider: unknown keys in the projection result. "
                f"Allowed keys: {list(_MESSAGE_KEYS)}. Unknown: {unknown}"
            )
        content = expression_result.get("content")
        raw_calls = expression_result.get("tool_calls")
        if raw_calls is not None and not isinstance(raw_calls, list):
            raise RuntimeError("SystemOne provider: tool_calls must be a list.")
        tool_calls = list(raw_calls or [])
        reasoning = expression_result.get("reasoning")
        if expression_result.get("finish_reason"):
            finish_reason = str(expression_result["finish_reason"])
    else:
        raise RuntimeError(
            "SystemOne provider: the projection expression must return a str or a dict "
            f"(got {type(expression_result).__name__})."
        )

    calls: list[SimpleNamespace] = []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            raise RuntimeError("SystemOne provider: each tool_calls entry must be a dict.")
        fn = tc.get("function") or {}
        arguments = fn.get("arguments", tc.get("arguments"))
        calls.append(
            SimpleNamespace(
                id=str(tc.get("id") or f"call_{i}"),
                type="function",
                function=SimpleNamespace(
                    name=str(fn.get("name") or tc.get("name") or ""),
                    arguments=arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False),
                ),
            )
        )

    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    usage_obj = SimpleNamespace(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=calls or None,
        reasoning_content=reasoning,
        refusal=None,
    )
    return SimpleNamespace(
        id=f"systemone-{model}",
        model=model,
        usage=usage_obj,
        choices=[SimpleNamespace(index=0, message=message, finish_reason=finish_reason)],
    )


# --- client ------------------------------------------------------------------

class SystemOneClient:
    """OpenAI-SDK-shaped client over the System One API (POST /v1/systemone).

    Hermes sees the usual `client.chat.completions.create(**kwargs)` surface;
    state assembly, question build, and answer-to-message conversion happen
    inside this client.
    """

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, *, api_key: str = "", base_url: Any = None, settings: dict[str, Any] | None = None,
                 default_headers: Any = None, timeout: Any = None, http_client: Any = None, **_: Any) -> None:
        import httpx

        self.api_key = api_key
        self.base_url = str(base_url or "").strip().rstrip("/")
        self._settings = settings or {}
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
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        response = self._http.post(url, json={"model": model, "state": state, "questions": questions},
                                   headers=headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"SystemOne API error: HTTP {response.status_code}: {response.text[:300]}"
            )
        data = response.json()
        expression = (settings.get("response") or {}).get("expression")
        answers = data.get("answers") or {}
        result = _eval_expression(str(expression).strip(), answers)
        return _assemble(result, answers, data.get("usage") or {}, model)


# --- transport ---------------------------------------------------------------

class SystemOneTransport(ProviderTransport):
    """`api_mode='systemone'` transport.

    Message conversion happens in the client, so this layer is identity. """

    @property
    def api_mode(self) -> str:
        return "systemone"

    def convert_messages(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return messages

    def convert_tools(self, tools: list[dict[str, Any]] | None) -> Any:
        return tools or []

    def build_kwargs(self, model: str, messages: list[dict[str, Any]], tools: Any = None, **params: Any) -> dict:
        return {"model": model, "messages": messages}

    def normalize_response(self, response: Any, **kwargs: Any) -> NormalizedResponse:
        choice = response.choices[0]
        msg = choice.message
        calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ] or None
        return NormalizedResponse(
            content=msg.content,
            tool_calls=calls,
            finish_reason=choice.finish_reason,
            reasoning=getattr(msg, "reasoning_content", None),
            usage=Usage.from_openai(response.usage) if getattr(response, "usage", None) else None,
        )


# --- provider profile --------------------------------------------------------

class SystemOneProfile(ProviderProfile):
    def create_client(self, **client_kwargs: Any) -> Any:
        allowed = {"api_key", "base_url", "timeout", "http_client", "default_headers"}
        return SystemOneClient(
            settings=_settings_for(self.name),
            **{key: value for key, value in client_kwargs.items() if key in allowed},
        )

    def fetch_models(self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0) -> list[str] | None:
        url = _v1_base(base_url or self.base_url) + "/models"
        import urllib.request

        request = urllib.request.Request(url)
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            return None
        items = data.get("data") if isinstance(data, dict) else data
        if not isinstance(items, list):
            return None
        return [str(item.get("id")) for item in items if isinstance(item, dict) and item.get("id")]


register_transport("systemone", SystemOneTransport)


# --- override of the bundled "custom" profile --------------------------------
# A `providers.<key>` entry with api_mode: systemone resolves through the custom
# runtime, whose provider name is "custom"; the bundled CustomProfile does not
# implement create_client, so no client is served for it. This override appends
# create_client: it returns the SystemOneClient only when the entry behind the
# URL declares api_mode: systemone, and None otherwise (which keeps the standard
# OpenAI client for every other route). The endpoint list lives in config.


class SystemOneCustomProfile(CustomBaseProfile):
    """Override of the bundled ``custom`` profile: serves the SystemOne client for
    ``api_mode: systemone`` routes."""

    def create_client(self, **client_kwargs: Any) -> Any:
        settings = _settings_for_base_url(client_kwargs.get("base_url"))
        if settings is None:
            return None
        allowed = {"api_key", "base_url", "timeout", "http_client", "default_headers"}
        return SystemOneClient(
            settings=settings,
            **{key: value for key, value in client_kwargs.items() if key in allowed},
        )


def _settings_for_base_url(base_url: Any) -> dict[str, Any] | None:
    """Find the `providers:` entry whose URL matches and which declares
    `api_mode: systemone`, else None (no client override). Runs at create_client
    time, when hermes_cli is already loaded; no scan happens at import time.
    """
    from hermes_cli.config import load_config

    url = str(base_url or "").strip().rstrip("/")
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


def _entry_url(entry: dict[str, Any]) -> str:
    url_str = str(entry.get("api") or entry.get("base_url") or entry.get("url") or "").strip().rstrip("/")
    return url_str if url_str.endswith("/v1") else url_str + "/v1"


register_provider(SystemOneCustomProfile(
    name="custom",
    aliases=(),  # local-server alias names already map to custom in the core alias table
    env_vars=(),
    base_url="",
))
