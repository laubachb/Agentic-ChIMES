---
name: chimes-open-models
description: Run the ChIMES agents on an open-weight LLM instead of a hosted API, through EZRA - check hardware, pick a model that fits, download it to shared scratch, and serve it on a login node, inside an idev/salloc allocation, or as an sbatch job. Use when the user asks for open-source / local / self-hosted models, has no API access on the cluster, or asks what model their node can run.
---

# Open-weight models for the agents (EZRA)

`chimes-agent` calls EZRA (https://github.com/Awwal41/EZRA) as an external
CLI for the hardware probe, model gallery and downloads. The model is then
served by an engine with OpenAI tool calling: llama.cpp `llama-server`
(GGUF, CPU or GPU) or `vllm` (safetensors, GPU). `ezra serve` is a
chat-only fallback and cannot drive tools.

## 1. Size it

```bash
chimes-agent llm-recommend                          # this host (login node is capped at 8 GB / 4 threads)
chimes-agent llm-recommend --for-machine dane       # one compute node from the profile's node: block
chimes-agent llm-recommend --memory-gb 512 --gpus 4 --gpu-memory-gb 80   # a GPU node the profile does not describe
```

Report `hardware.budget_gb`, the top picks with `need_gb`, and
`engines_on_path`. The need includes the KV cache for `--ctx-size`
(default 32k tokens; agent system prompts are long, do not go below 16k).

## 2. Download (login node)

Compute nodes often have no internet. Download from the login node into
shared scratch so jobs can read it. Ask the user first: it is several GB.

```bash
chimes-agent llm-download --model qwen2.5-7b-instruct --for-machine dane --dry-run
chimes-agent llm-download --model qwen2.5-7b-instruct --for-machine dane   # run in the background
```

Gated models need `HF_TOKEN` and an accepted license; ask the user, do not retry.

## 2b. Agent profile (optional)

`chimes-agent llm-agent --for-machine <profile>` installs EZRA's
**chimes-assistant** agent (a prompt condensed from CLAUDE.md, plus preferred
/ fallback models re-picked for that node). Afterwards
`llm-serve --agent <name>` serves the first downloaded model on
its list. Use it when the user runs a harness other than Claude Code, or
wants `ezra chat --agent` / the EZRA Grid.

## 3. Serve (one of three places)

| where | when | command |
|---|---|---|
| login node | a quick try with a small model (≤ 8 GB) | `llm-serve --work-dir <dir> --model <m>` |
| interactive allocation | the user has an `idev` / `salloc` shell | same command, run inside that shell (`--where allocation` is automatic) |
| batch job | anything larger, or for hours | `llm-serve --work-dir <shared dir> --model <m> --machine <profile> --dry-run`, then without `--dry-run` after approval |

GPU batch jobs: `--gpus N --queue <gpu partition>`. Batch work dirs must be
on the shared filesystem (same rule as every job).

Then poll `chimes-agent llm-serve --status --work-dir <dir>` (or delegate
to `chimes-job-monitor`) until `state` is `ready`. It returns `base_url`,
`client.exports` (OpenAI variables; with llama-server also the Anthropic
ones Claude Code reads), `client.cline_auth` for Cline, and an
`ssh_tunnel` line for a laptop. Stop with
`--stop` when the user is done; a batch server holds a node until then.

## Be straight with the user

- Small open models (≤ 14B) follow long multi-step tool workflows far less
  reliably than frontier models. Suggest a 30B+ model on a GPU node for the
  orchestrator, and keep a human in the loop on every approval.
- The approval hook in `.claude/hooks/` only runs inside Claude Code. Other
  harnesses (OpenCode, Aider, Cline) need their own permission prompts on;
  keep the dry-run-first rule either way.
- The API key in `<work-dir>/api_key` guards the server from other users of
  the cluster. Never print it; use the `$(cat ...)` form from `client.exports`.
