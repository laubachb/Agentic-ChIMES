# `chimes-agent llm-recommend`

**Status: implemented.** Read-only; needs [EZRA](https://github.com/Awwal41/EZRA).

Which open-weight chat model can drive the agents on a given machine. See
[Open-weight models](../guide/open_models.md) for the whole workflow.

```bash
chimes-agent llm-recommend                                  # this host, probed by EZRA
chimes-agent llm-recommend --for-machine dane               # one Dane compute node (profile node: block)
chimes-agent llm-recommend --memory-gb 512 --gpus 4 --gpu-memory-gb 80 --keyword qwen
```

The hardware comes from `ezra models recommend --json` (RAM, CPU count,
NVIDIA GPUs), from the `node:` block of a [machine profile](../concepts/machine_profiles.md)
with `--for-machine`, or from the flags, which override either.

| context | memory budget |
|---|---|
| GPU node | GPUs × VRAM per GPU |
| CPU node, workstation | 80% of RAM |
| login node (outside a Slurm job on a cluster) | the above, capped at 8 GB: login nodes are shared |

Candidates are EZRA's gallery chat models (encoders and gated models are
left out unless `--include-gated`). A model fits when its weights plus the
KV cache for `--ctx-size` tokens (default 32768) fit the budget:

- weights: the gallery's `recommended_ram_gb` when EZRA reports it, else
  ~0.6 GB per billion parameters for GGUF and ~2 GB for 16-bit safetensors;
- KV cache: ~0.25 GB per billion parameters per 32k tokens (active
  parameters for mixture-of-experts models, e.g. 17 for "109B-A17B").

Fitting models are ranked largest first, since larger models follow
multi-step tool instructions more reliably. General instruct models win
ties over domain chat models, reasoning distills count half, and
safetensors need a GPU.

Output: `hardware` (with `budget_gb` and `context`), `engines_on_path`
(`llama-server`, `vllm`, `ezra`), `recommendations` (each with `need_gb`,
`engine`, `reasons`), `top_pick`, `next` (the download command) and `notes`.
