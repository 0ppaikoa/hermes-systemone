"""Response assembly for the System One provider plugin.

System One `answers` -> the OpenAI-shaped response Hermes expects, through a
user-provided projection expression (`models.<task>.response.expression`).
"""

from __future__ import annotations

import ast as _ast
import json
from types import SimpleNamespace
from typing import Any

_MESSAGE_KEYS = ("content", "tool_calls", "reasoning", "finish_reason")

_FINISH_REASONS = ("stop", "tool_calls", "length", "content_filter")


def _int_or_zero(value: Any) -> int:
    """Non-negative int with a safe fallback (provider usage numbers may be junk)."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _eval_expression(expression: str, answers: dict[str, Any]) -> Any:
    """Evaluate the projection expression with simpleeval. Namespaces: `answers`
    (plain data) plus a whitelist of builtins. Design doc 5.1.

    simpleeval's default evaluator does not evaluate dict / list literals
    (FeatureNotAvailable), and the projection expression returns a dict
    (design doc 5.2), so exactly those nodes are added. Comprehensions stay
    rejected.
    """
    import simpleeval  # optional dependency: imported lazily so a missing install
    # still lets plugin discovery register transport + profiles

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
        candidate = str(expression_result.get("finish_reason") or "")
        if candidate in _FINISH_REASONS:
            finish_reason = candidate
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

    usage = usage if isinstance(usage, dict) else {}
    input_tokens = _int_or_zero(usage.get("input_tokens"))
    output_tokens = _int_or_zero(usage.get("output_tokens"))
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
