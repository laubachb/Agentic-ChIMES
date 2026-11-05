# `chimes-agent llm-download`

**Status: implemented.** Needs [EZRA](https://github.com/Awwal41/EZRA). Asks for approval (several GB of disk).

Download open-weight model(s) from EZRA's gallery.

```bash
chimes-agent llm-download --model qwen2.5-7b-instruct --for-machine dane --dry-run
chimes-agent llm-download --model qwen2.5-7b-instruct --for-machine dane
chimes-agent llm-download --model qwen2.5-7b-instruct --models-dir /p/lustre2/$USER/models --quant Q4_K_M
```

Run it on a login node: many clusters give compute nodes no internet. With
`--for-machine`, weights go to `<scratch_root>/ezra/models` of that profile,
which batch jobs can read; node-local directories (`/tmp`, ...) are
refused. Without either flag EZRA's own default (`~/.ezra/models` or
`EZRA_MODELS_DIR`) is used. Pass the same `--models-dir` (or
`--machine`) to `llm-serve`.

`--dry-run` passes through to `ezra models download --dry-run`: it lists
what would be fetched and writes nothing.

Gated models (Llama, Gemma, ...) need `HF_TOKEN` and the license accepted
on Hugging Face; a 401 comes back with that fix.

Output: `models_dir`, EZRA's per-model `results` and, per model, whether it
is `installed`, its `path`, `format`, `bytes_on_disk` and `license`.
