"""System 1 wire provider plugin for Hermes Agent.

System 1 は判定特化 wire。TypeSafe 社が定義し、TypeSafe 公式と Ollama Cloud などが
POST /v1/systemone で提供する。テキストを生成しないため、このプラグインは:

- Hermes の履歴（OpenAI 形）を `state` に畳む。書式は `providers.<name>.state`
- 質問を設定 `providers.<name>.questions` から組み立てる。省略はエラー
- 回答 `answers` を射影式 `providers.<name>.response.expression` でアシスタント
  メッセージに組み立て直す。省略はエラー

コアの改造はしない。設計書: systemone-provider-design.md（AokiApp、2026-11-09）。
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

WIRE_PATH = "/systemone"


def _v1_base(base: Any) -> str:
    """base_url を /v1 で終わる形に正規化する。`https://ollama.com` → `https://ollama.com/v1`。"""
    base = str(base or "").strip().rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"

# --- 設定の読み取り ----------------------------------------------------------


def _settings_for(provider: str) -> dict[str, Any]:
    """`providers.<name>` または `custom_providers:[]` から *provider* の設定を取り出す。

    見つからなければ空 dict。質問と射影式の省略は生成時にエラーにする。
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
                "SystemOne provider: 質問セットが設定されていません。"
                "providers.<name>.questions に System 1 の questions を書いてください。"
            )
        return
    expression = (settings.get("response") or {}).get("expression")
    if not (isinstance(expression, str) and expression.strip()):
        raise RuntimeError(
            "SystemOne provider: 射影式が設定されていません。"
            "providers.<name>.response.expression を書いてください。"
        )


# --- state の畳み込み ---------------------------------------------------------

_ROLES = ("system", "user", "assistant", "tool", "reasoning")


def _escape(text: Any) -> str:
    """内容を JSON エスケープする（1行保証）。外側の引用符は剥がす。"""
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
    """tool_call_id → (name, arguments)。tool 結果行を組み立てるのに使う。"""
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


def _fold_state_string(messages: list[dict[str, Any]], includes: dict[str, Any]) -> str:
    """OpenAI 形 messages → `role: 内容` の行書式。設計書 §4.1。"""
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


def _fold_state_array(messages: list[dict[str, Any]], includes: dict[str, Any]) -> list[dict[str, Any]]:
    """OpenAI 形 messages → includes で選ばれた要素を残した配列。設計書 §4.2。

    role ラベルは必須（配列の要素が role を運ぶ）。
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


def _fold_state(messages: list[dict[str, Any]], settings: dict[str, Any]) -> Any:
    state_cfg = settings.get("state") or {}
    fmt = str(state_cfg.get("format") or "string").strip().lower()
    includes = state_cfg.get("includes")
    includes = includes if isinstance(includes, dict) else {}
    if fmt == "array":
        return _fold_state_array(messages, includes)
    return _fold_state_string(messages, includes)


# --- 応答の組み立て -----------------------------------------------------------

def _eval_expression(expression: str, answers: dict[str, Any]) -> Any:
    """射影式を simpleeval で評価する。名前空間は answers（plain data）と
    ホワイトリスト済みのビルトインだけ。設計書 §5.1。

    simpleeval の標準 evaluator は dict / list リテラルを評価しない（実測:
    FeatureNotAvailable）。射影式は dict を返す（§5.2）ので、その2ノードだけを
    最小限足した派生 evaluator を使う。内包表記（comprehension）は許可しない。
    """
    import ast as _ast

    import simpleeval

    functions = {
        "len": len,
        "round": round,
        "int": int,
        "float": float,
        "str": str,
        # 書式化。文字列メソッド（"%.2f".format(x)）は simpleeval が禁止するため、
        # Python ビルトインの 2引数 format(value, spec) をホワイトリストで提供する。
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
    """射影式の結果（Union[dict, str]）→ OpenAI 形 response。設計書 §5.2 / §5.3。"""
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
                "SystemOne provider: 射影式の戻り値に未知のキーがあります。"
                f"許容キーは {list(_MESSAGE_KEYS)}。未知: {unknown}"
            )
        content = expression_result.get("content")
        raw_calls = expression_result.get("tool_calls")
        if raw_calls is not None and not isinstance(raw_calls, list):
            raise RuntimeError("SystemOne provider: tool_calls には list を返してください。")
        tool_calls = list(raw_calls or [])
        reasoning = expression_result.get("reasoning")
        if expression_result.get("finish_reason"):
            finish_reason = str(expression_result["finish_reason"])
    else:
        raise RuntimeError(
            "SystemOne provider: 射影式は str または dict を返してください。"
            f"（実際: {type(expression_result).__name__}）"
        )

    wire_calls = []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            raise RuntimeError("SystemOne provider: tool_calls の各要素は dict にしてください。")
        fn = tc.get("function") or {}
        arguments = fn.get("arguments", tc.get("arguments"))
        wire_calls.append(
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
        tool_calls=wire_calls or None,
        reasoning_content=reasoning,
        refusal=None,
    )
    return SimpleNamespace(
        id=f"systemone-{model}",
        model=model,
        usage=usage_obj,
        choices=[SimpleNamespace(index=0, message=message, finish_reason=finish_reason)],
    )


# --- クライアント façade -------------------------------------------------------

class SystemOneClient:
    """OpenAI-SDK 形の窓口。裏で System 1 wire（POST /v1/systemone）を叩く。

    Hermes からは `client.chat.completions.create(**kwargs)` で見える。
    質問と state の組み立て、answers → アシスタントメッセージへの組み立て直しは
    この窓口の裏で完結する。
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
                "SystemOne provider: base_url が設定されていません。"
                "providers.<name>.base_url に System 1 実装の URL を書いてください。"
            )
        return _v1_base(base) + WIRE_PATH

    def close(self) -> None:
        self.is_closed = True
        try:
            self._http.close()
        except Exception:
            pass

    _require_settings = staticmethod(_require_settings)

    def _create(self, *, model: str = "", messages: Any = None, stream: bool = False, **kwargs: Any) -> Any:
        if stream:
            raise RuntimeError("SystemOne provider: 現設計は非ストリーミングのみです（supports_streaming=False）。")
        settings = self._settings
        _require_settings(settings, part="questions")
        _require_settings(settings, part="expression")
        questions = settings.get("questions") or {}
        state = _fold_state(list(messages or []), settings)
        url = self._url()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        response = self._http.post(url, json={"model": model, "state": state, "questions": questions},
                                   headers=headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"SystemOne wire error: HTTP {response.status_code}: {response.text[:300]}"
            )
        data = response.json()
        expression = (settings.get("response") or {}).get("expression")
        answers = data.get("answers") or {}
        result = _eval_expression(str(expression).strip(), answers)
        return _assemble(result, answers, data.get("usage") or {}, model)


# --- トランスポート ------------------------------------------------------------

class SystemOneTransport(ProviderTransport):
    """api_mode='systemone' のトランスポート。

    メッセージから wire 形への翻訳は façade 側で完結するので、この層は identity。"""

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


# --- プロバイダプロファイル ------------------------------------------------------

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
register_provider(SystemOneProfile(
    name="systemone",
    aliases=("s1",),
    api_mode="systemone",
    env_vars=("SYSTEMONE_API_KEY",),
    base_url="",  # 設定から取得（設計書 §2 拘束条件5）
    auth_type="api_key",
    supports_streaming=False,
    fallback_models=(),
))
