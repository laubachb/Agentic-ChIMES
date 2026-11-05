# `chimes-agent llm-agent`

**Status: implemented.** Needs [EZRA](https://github.com/Awwal41/EZRA). Writes a few small files.

Install an [EZRA personal agent](https://github.com/Awwal41/EZRA/blob/main/docs/agents.md)
that knows how to drive `chimes-agent`, with its models picked for the
machine it will run on.

```bash
chimes-agent llm-agent                                   # models for this host
chimes-agent llm-agent --for-machine dane --replace      # re-pick for a Dane compute node
chimes-agent llm-agent --name local/my-chimes --template ./my-agent-folder
```

An EZRA agent is a folder with three files:

| file | contents |
|---|---|
| `agent.yaml` | `models.preferred` / `models.fallback` (gallery names), `tags`, `science_focus`, `prompt` |
| `prompt.md` | the system prompt: role, priorities, typical tasks, style, boundaries |
| `NOTES.md` | notes for people (owner, intended use, hardware) |

This package ships `chimes-assistant`
(`src/agentic_chimes/ezra_agents/chimes-assistant/`), whose prompt condenses
the rules in `CLAUDE.md`: `--describe` before use, one JSON object per
call, dry run and approval before any submission, one level of theory per
fit, shared filesystem for jobs, no invented numbers.

The stage follows EZRA's own steps (`ezra agents init`, `models recommend`,
`agents check`):

1. copies the template to `<ezra-root>/agents/<name>/`. The root defaults
   to `CHIMES_EZRA_ROOT` or `~/.chimes-agent/ezra` and is passed to EZRA as
   `EZRA_ROOT`, so your EZRA checkout is not modified;
2. re-picks the models with [`llm-recommend`](llm-recommend.md)'s ranking for
   this host or `--for-machine`: the two strongest fits become `preferred`,
   the two strongest 3-9B fits become `fallback` (`--keep-models` keeps the
   template's lists);
3. checks every alias against the gallery and that `ezra agents show`
   loads the profile.

Output: `path`, `prompt`, `preferred`, `fallback`, `hardware`, `check`
(`unknown_aliases`, `not_downloaded`, `ezra_load_error`) and `next`.

Then [`llm-serve --agent <name>`](llm-serve.md) serves the first downloaded
model on the list. With the `ezra` engine the prompt is applied server-side
(model id `agent:<name>`); with `llama-server` or `vllm` the harness supplies
the prompt (Claude Code reads `CLAUDE.md`; for Cline, put `prompt.md` in
`.clinerules`).
