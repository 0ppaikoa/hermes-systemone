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

One expression per endpoint. It maps the answers to the assistant message: a
string becomes the message content; a dict may set `content`, `tool_calls`,
`reasoning` and `finish_reason`.

More endpoints: add more entries. `api_mode: systemone` selects this protocol, so
the plugin never changes.

## Config reference

| Key | Required | Default | Notes |
|---|---|---|---|
| `api` | yes | — | endpoint URL |
| `key_env` | yes | — | env variable holding the API key |
| `default_model` | yes | — | model name at the endpoint |
| `api_mode` | yes | `chat_completions` | set to `systemone` |
| `state.format` | no | `string` | `string` or `array` |
| `state.includes.*` | no | all `false` | include more history; the latest user message is always sent |
| `questions` | yes | — | System One questions, sent as-is |
| `response.expression` | yes | — | Python expression; may return `str` or `dict` |
| dict → `content` | no | `null` | message text |
| dict → `tool_calls` | no | `null` | list of `{name, arguments}` dicts |
| dict → `reasoning` | no | `null` | free text |
| dict → `finish_reason` | no | `"stop"` | |

## License

MIT — see [LICENSE](LICENSE).
