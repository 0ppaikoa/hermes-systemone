# hermes-systemone

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) model-provider plugin
for the **System 1** decision API — a non-generative protocol served at
`POST /v1/systemone` by several vendors and self-hosted servers.

System 1 models answer typed questions about a `state` — yes/no with a
probability, a choice among criteria, a score on a rubric. They never generate
text. This plugin makes one usable as the main-loop model in Hermes: it sends
the conversation as `state`, asks the questions from config, and turns the
answers into an assistant message.

## Install

```bash
hermes -p <profile> plugins install 0ppaikoa/hermes-systemone
```

Needs [simpleeval](https://pypi.org/project/simpleeval/).

## Config

One `providers:` entry per endpoint, with `api_mode: systemone`. Each task lives
under `models.<name>`:

```yaml
providers:
  systemone:
    api: https://your-system-one-endpoint/v1
    key_env: MY_SYSTEMONE_API_KEY
    api_mode: systemone
    discover_models: false        # required — see below
    default_model: router         # the task used when the caller names none
    models:
      router:
        base_model: my-model      # the id the endpoint serves
        state:
          format: string          # string | array
          includes:
            system: true
            user: true
            assistant: true
            tool: true
            reasoning: false
        questions:
          next_action:
            type: choice
            instructions: "What should the agent do next?"
            criteria:
              reply: "Reply to the user now"
              call_tool: "Invoke a tool"
              done: "The task is complete"
        response:
          expression: >-
            'Replying to the user' if answers.next_action.choice == 'reply'
            else 'Task complete' if answers.next_action.choice == 'done'
            else 'Invoking a tool'
```

The key under `models` is the name callers pass as `model=`; `base_model` is the
id sent to the endpoint. Several tasks may share one `base_model`, so a router
and a guard can run on the same model with different questions:

```yaml
    models:
      router:
        base_model: my-model
        questions: { next_action: { ... } }
        response: { expression: "..." }
      guard:
        base_model: my-model      # same model, different questions
        questions: { harm: { ... } }
        response: { expression: "..." }
```

`model=router` and `model=guard` reach the two tasks. A name that is not under
`models` raises instead of answering with another task's questions, and so does a
task without `base_model`.

### `discover_models: false` is required

Hermes prefers an endpoint's live `/v1/models` listing over a configured `models`
mapping unless the mapping is an allowlist — and a mapping is not one. With
discovery left on, the task names under `models` never reach the picker. The
plugin refuses to build a client without `discover_models: false` rather than let
that surface later as an unselectable model.

More endpoints: add more entries. `api_mode: systemone` selects this protocol, so
the plugin never changes.

## Auxiliary tasks

A task under `models` can serve an auxiliary task — approval review, compression,
titles. Name the entry as the provider and the task as the model:

```yaml
auxiliary:
  approval:
    provider: my-decider     # the providers: key
    model: guard             # the task under models
    timeout: 30
```

Leave `base_url`, `api_key` and `key_env` out. Hermes collapses a task whose
config carries both `base_url` and `api_key` to the provider name `custom`
(`agent/auxiliary_client.py::_resolve_task_provider_model`), which erases the
entry name before the plugin is consulted. The entry already holds the endpoint
and the credential, so the task does not need them.

The plugin registers one profile per `api_mode: systemone` entry, under the
entry's own name. That registration is what makes the auxiliary path work: it
resolves a task's provider by name through `PROVIDER_REGISTRY` and calls that
profile's `create_client()`. Without it the name is unknown there and the task
falls through to a plain OpenAI client, which posts to `/chat/completions` — a
path a System One endpoint does not serve.

The profile's `env_vars` comes from the entry's `key_env`. It must not be empty:
`PROVIDER_REGISTRY` only holds providers that declare a credential variable, so
a profile declaring none is invisible to the resolver.

## Config reference

| Key | Required | Default | Notes |
|---|---|---|---|
| `api` | yes | — | endpoint URL |
| `key_env` | yes | — | env variable holding the API key |
| `api_mode` | yes | `chat_completions` | **set to `systemone`**; otherwise the entry is served by the standard OpenAI client. |
| `discover_models` | yes | `true` | **must be `false`** — see `discover_models: false` is required |
| `default_model` | no | — | the task used when the caller passes no `model`. |
| `models.<name>.base_model` | yes | — | model id sent to the endpoint |
| `models.<name>.state.format` | no | `string` | how the thread is provided — see State format |
| `models.<name>.state.includes.*` | no | all `false` | include more history; the latest user message is always sent |
| `models.<name>.questions` | yes | — | [System One questions](https://docs.typesafe.ai/api/system-one) |
| `models.<name>.response.expression` | yes | — | see Response expression |

## State format

`state.format` specifies how the thread is provided to the System 1 model.

**`string`** (default) renders the thread as one `role: JSON-escaped content`
line per message — what the model sees looks like:

```
system: You are a helpful router.
user: I was charged twice.
tool: get_charges({"user": "me"}) => {"count": 2, "amounts": [1200, 1200]}
assistant: Found it — refunding.
user: Thanks. Please also email me a receipt.
```

**`array`** passes the thread as the OpenAI-style list of messages
(`role` / `content` / `tool_calls` / `tool_call_id`), the same shapes the
OpenAI API uses.

## Response expression

`models.<name>.response.expression` is a Python expression, evaluated with
[simpleeval](https://pypi.org/project/simpleeval/) on every turn. `answers` holds
the System One response keyed by question name. What the expression returns is
what becomes the assistant message:

- **A string** becomes the message content, and the turn ends there (no tool
  calls).
- **A dict** sets the message fields explicitly: `content` is the message text;
  `tool_calls` is a list of `{name, arguments}` dicts the agent runs next;
  `reasoning` is free text; `finish_reason` defaults to `"stop"`.

The `questions` body and its answer shapes follow the System One API definition:
[System One API reference](https://docs.typesafe.ai/api/system-one).

## Streaming

The System One API has no streaming form. The profile declares
`supports_streaming = False` (Hermes' standard opt-out), so the main loop runs
the exchange non-streaming from the first attempt — no per-turn fallback delay,
and no streaming retry path that could hand the client header values the
plain-httpx facade cannot represent.

## License

MIT — see [LICENSE](LICENSE).
