"""Tests for the ``decide`` tool and its two JSON Schemas (stdlib unittest)."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import types
import unittest
from unittest import mock

import jsonschema

ROOT = pathlib.Path(__file__).resolve().parents[1]
if "hermes_systemone" not in sys.modules:
    package = types.ModuleType("hermes_systemone")
    package.__path__ = [str(ROOT)]
    sys.modules["hermes_systemone"] = package

from hermes_systemone import decide, tool_config  # noqa: E402

TOOL = decide.TOOL_PARAMETERS
STRICT = decide.STRICT_SCHEMA

VALID = {
    "noul": {"state": "x", "questions": {"a": {"type": "noul", "instructions": "q"}}},
    "noul_criteria": {
        "state": "x",
        "questions": {"a": {"type": "noul", "instructions": "q",
                            "criteria": {"true": "y", "false": "n"}}},
    },
    "choice": {
        "state": {"cmd": "ls"},
        "questions": {"tone": {"type": "choice", "instructions": "tone?",
                               "criteria": {"angry": "upset", "calm": "neutral"}}},
    },
    "score": {
        "state": ["a", "b"],
        "questions": {"urgency": {"type": "score", "instructions": "urgent?",
                                  "criteria": ["low", "high"]}},
    },
}

INVALID = {
    "choice_one_option": {"state": "x", "questions": {"t": {"type": "choice", "instructions": "q",
                                                            "criteria": {"a": "A"}}}},
    "score_one_level": {"state": "x", "questions": {"u": {"type": "score", "instructions": "q",
                                                          "criteria": ["a"]}}},
    "noul_one_side": {"state": "x", "questions": {"a": {"type": "noul", "instructions": "q",
                                                        "criteria": {"true": "y"}}}},
    "null_criterion": {"state": "x", "questions": {"t": {"type": "choice", "instructions": "q",
                                                         "criteria": {"a": None, "b": "B"}}}},
    "duplicate_level": {"state": "x", "questions": {"u": {"type": "score", "instructions": "q",
                                                          "criteria": ["a", "a"]}}},
    "blank_instructions": {"state": "x", "questions": {"a": {"type": "noul", "instructions": "  "}}},
    "bad_name": {"state": "x", "questions": {"Tone": {"type": "noul", "instructions": "q"}}},
    "unknown_type": {"state": "x", "questions": {"a": {"type": "box", "instructions": "q"}}},
}


def ok(schema, instance):
    return not list(jsonschema.Draft202012Validator(schema).iter_errors(instance))


def config():
    return {
        "endpoints": {"ollama": {"api": "https://ollama.com/v1", "key_env": "TEST_KEY"}},
        "models": {"clef27": {"endpoint": "ollama", "base_model": "clef:27b"}},
        "default_model": "clef27",
    }


class _Response:
    status_code = 200
    text = '{"answers": {"a": {"type": "noul", "noul": 0.9}}}'

    def json(self):
        return {"answers": {"a": {"type": "noul", "noul": 0.9}}}


class _Client:
    calls: list = []

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return _Response()


class SchemaTests(unittest.TestCase):
    def test_both_schemas_are_valid(self):
        jsonschema.Draft202012Validator.check_schema(TOOL)
        jsonschema.Draft202012Validator.check_schema(STRICT)

    def test_valid_examples_pass_both(self):
        for name, instance in VALID.items():
            with self.subTest(name=name):
                self.assertTrue(ok(TOOL, instance))
                self.assertTrue(ok(STRICT, instance))

    def test_invalid_examples_fail_strict(self):
        for name, instance in INVALID.items():
            with self.subTest(name=name):
                self.assertFalse(ok(STRICT, instance))

    def test_tool_schema_catches_shape_errors(self):
        for name in ("choice_one_option", "score_one_level", "noul_one_side", "unknown_type"):
            with self.subTest(name=name):
                self.assertFalse(ok(TOOL, INVALID[name]))

    def test_tool_schema_leaves_fine_rules_to_strict(self):
        for name in ("duplicate_level", "blank_instructions", "bad_name"):
            with self.subTest(name=name):
                self.assertTrue(ok(TOOL, INVALID[name]))


class ValidationTests(unittest.TestCase):
    def test_state_rejects_empty_and_non_json(self):
        for state in ("", "   ", {}, [], 3, True, None):
            with self.subTest(state=state):
                with self.assertRaises(decide.DecideError):
                    decide.validate_state(state)

    def test_state_size_limit(self):
        with self.assertRaises(decide.DecideError):
            decide.validate_state("x" * (decide._MAX_STATE_CHARS + 1))

    def test_question_error_has_path_and_example(self):
        with self.assertRaises(decide.DecideError) as ctx:
            decide.validate_questions(
                {"tone": {"type": "choice", "instructions": "q", "criteria": {"a": "A"}}}
            )
        message = str(ctx.exception)
        self.assertIn("questions.tone.criteria", message)
        self.assertIn("2件以上", message)
        self.assertIn("例:", message)

    def test_question_name_must_be_snake_case(self):
        with self.assertRaises(decide.DecideError) as ctx:
            decide.validate_questions({"Tone": {"type": "noul", "instructions": "q"}})
        self.assertIn("snake_case", str(ctx.exception))


class ConfigTests(unittest.TestCase):
    def test_accepts_the_documented_shape(self):
        cfg = tool_config.load_tool_config({
            "plugins": {"entries": {"hermes-systemone": {
                "endpoints": {"ollama": {"api": "https://ollama.com/v1", "key_env": "K"}},
                "models": {"clef27": {"endpoint": "ollama", "base_model": "clef:27b"}},
                "default_model": "clef27",
            }}}
        })
        self.assertEqual(cfg["models"]["clef27"]["base_model"], "clef:27b")
        self.assertEqual(cfg["default_model"], "clef27")

    def test_rejects_bad_entries(self):
        bad = [
            {"endpoints": {"ollama": {"api": "https://ollama.com/v1", "key_env": "K", "x": 1}},
             "models": {}},
            {"endpoints": {"ollama": {"api": "https://ollama.com/v1", "key_env": "K"}},
             "models": {"clef27": {"endpoint": "nope", "base_model": "clef:27b"}}},
            {"endpoints": {"ollama": {"api": "https://ollama.com/v1", "key_env": "K"}},
             "models": {"clef27": {"endpoint": "ollama", "base_model": "clef:27b"}},
             "default_model": "missing"},
        ]
        for entry in bad:
            with self.subTest(entry=entry):
                with self.assertRaises(tool_config.ToolConfigError):
                    tool_config.load_tool_config({"plugins": {"entries": {"hermes-systemone": entry}}})

    def test_resolve_model_uses_default_and_rejects_unknown(self):
        cfg = config()
        self.assertEqual(tool_config.resolve_model(cfg, None)[0], "clef27")
        with self.assertRaises(tool_config.ToolConfigError):
            tool_config.resolve_model(cfg, "unknown")


class HandlerTests(unittest.TestCase):
    def setUp(self):
        _Client.calls.clear()

    def test_posts_the_body_and_returns_answers(self):
        with mock.patch.object(decide.httpx, "Client", _Client), \
                mock.patch.object(decide, "load_tool_config", config), \
                mock.patch.dict(os.environ, {"TEST_KEY": "secret-key"}):
            result = decide.decide({
                "state": "hello",
                "questions": {"a": {"type": "noul", "instructions": "q"}},
            })
        self.assertEqual(json.loads(result), {"a": {"type": "noul", "noul": 0.9}})
        call = _Client.calls[0]
        self.assertEqual(call["url"], "https://ollama.com/v1/systemone")
        self.assertEqual(call["headers"]["Authorization"], "Bearer secret-key")
        self.assertEqual(call["json"], {
            "model": "clef:27b",
            "state": "hello",
            "questions": {"a": {"type": "noul", "instructions": "q"}},
        })

    def test_rejects_before_posting(self):
        with mock.patch.object(decide.httpx, "Client", _Client), \
                mock.patch.object(decide, "load_tool_config", config), \
                mock.patch.dict(os.environ, {"TEST_KEY": "secret-key"}):
            with self.assertRaises(decide.DecideError):
                decide.decide({"state": "hello", "questions": {"a": {"type": "noul"}}})
        self.assertEqual(_Client.calls, [])


if __name__ == "__main__":
    unittest.main()
