# `chimes-agent llm-serve`

**Status: implemented.** Asks for approval when it starts a server.

Serve an open-weight model behind an OpenAI-compatible API, on a login
node, inside an interactive allocation, or as a Slurm job.

```bash
# login node: small model, 127.0.0.1, 4 threads
chimes-agent llm-serve --work-dir ~/llm --model qwen2.5-3b-instruct

# inside `idev -p skx-dev -N 1 -t 02:00:00` (TACC) or `salloc -N 1 -p pdebug -t 2:00:00`
chimes-agent llm-serve --work-dir $SCRATCH/llm --model qwen2.5-7b-instruct

# a batch job: preview, then submit
chimes-agent llm-serve --work-dir /p/lustre2/$USER/llm --model qwen2.5-14b-instruct --machine dane --walltime-hours 8 --dry-run
chimes-agent llm-serve --work-dir /p/lustre2/$USER/llm --model qwen2.5-14b-instruct --machine dane --walltime-hours 8

chimes-agent llm-serve --status --work-dir /p/lustre2/$USER/llm
chimes-agent llm-serve --stop   --work-dir /p/lustre2/$USER/llm
```

## Where it runs

`--where auto` (the default) picks `batch` when `--machine` is given,
`allocation` inside a Slurm job (`SLURM_JOB_ID` set, as in an `idev` or
`salloc` shell), else `login`.

| `--where` | binds | threads | how |
|---|---|---|---|
| `login` | 127.0.0.1 | 4 (at most 8) | detached process on this node |
| `allocation` | 0.0.0.0 | the allocation's CPUs | in place when the shell is on the compute node, else `srun --overlap` onto it |
| `batch` | 0.0.0.0 | the profile's `node.cores` | one-node sbatch job through the same renderer as every stage; `--gpus N` adds `--gres=gpu:N` |

Every mode runs the same `<work-dir>/serve.sh`. It picks a free port (or
`--port`), writes `endpoint.runtime.json` (host, port, pid, job id), and
execs the engine. The server log is `<work-dir>/server.log`.

## Engines

| `--engine` | weights | tool calls | notes |
|---|---|---|---|
| `llama-server` | GGUF | yes (`--jinja`) | llama.cpp; CPU, or GPU with all layers offloaded when `--gpus` > 0 |
| `vllm` | safetensors | yes (`--enable-auto-tool-choice`, `--tool-call-parser`, default `hermes`) | GPU only |
| `ezra` | gallery names | no | `ezra serve`; plain chat, no API key |
| `auto` | | | `llama-server` for GGUF (else `ezra`), `vllm` for safetensors |

Pass `--engine-bin` when the binary is not on PATH. Weights come from an
EZRA gallery name (`--model`, downloaded with [`llm-download`](llm-download.md)),
an EZRA agent (`--agent`, the first downloaded model in its preferred then
fallback list; see [`llm-agent`](llm-agent.md)), or any `.gguf` file or
Hugging Face directory (`--model-path`). With `--agent` and the `ezra`
engine the server applies the agent's prompt and serves it as
`agent:<name>`; other engines report the prompt path for the harness.

## Access control

A random key is written to `<work-dir>/api_key` (mode 0600) and the server
requires it (`--api-key-file` for llama-server, `VLLM_API_KEY` for vllm),
so other users of a shared node cannot use your server. The key never
appears on a command line, where `ps` would show it.

## `--status`

| `state` | meaning |
|---|---|
| `queued` | batch job waiting in the queue |
| `starting` / `loading` | process up, model still loading (`log_tail` shows progress) |
| `ready` | answering `/v1/models` with the key |
| `stopped` | process or job gone |
| `none` | nothing was launched from this directory |

When ready it returns `base_url`, `served_models`, `client.exports`
(`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`; with llama-server also
`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL` for Claude
Code), `client.cline_auth` (the `cline auth --provider openai-compatible ...`
line EZRA's `cline use-local` would write, with this server's key and URL)
and, off the login node, an `ssh_tunnel` command for reaching it
from a laptop.

`--stop` kills the process group (login, allocation) or `scancel`s the job.
