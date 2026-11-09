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

Add a `providers:` entry with `api_mode: systemone`:

```yaml
providers:
  my-decider:
    api: https://your-system-one-endpoint/v1
    key_env: MY_SYSTEMONE_API_KEY
    default_model: my-model
    api_mode: systemone
    state:
      format: string           # string | array
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

More endpoints: add more entries. `api_mode: systemone` selects this protocol, so
the plugin never changes.

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

`providers.<id>.response.expression` is a Python expression, evaluated with
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

## Config reference

| Key | Required | Default | Notes |
|---|---|---|---|
| `api` | yes | — | endpoint URL |
| `key_env` | yes | — | env variable holding the API key |
| `default_model` | yes | — | model name at the endpoint |
| `api_mode` | yes | `chat_completions` | **set to `systemone`**; otherwise the entry is served by the standard OpenAI client. |
| `state.format` | no | `string` | how the thread is provided — see State format |
| `state.includes.*` | no | all `false` | include more history; the latest user message is always sent |
| `questions` | yes | — | [System One questions](https://docs.typesafe.ai/api/system-one) |
| `response.expression` | yes | — | see Response expression |

## License

MIT — see [LICENSE](LICENSE).
