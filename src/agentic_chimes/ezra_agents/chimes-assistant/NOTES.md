# chimes-assistant - notes for humans

- **Owner:** Agentic ChIMES maintainers.
- **Intended use:** an open-model counterpart of the Claude Code setup in
  `CLAUDE.md` and `.claude/`: drive `chimes-agent` stages with a human
  approving every submission.
- **Hardware target:** a CPU compute node (GGUF via llama-server) or a GPU
  node. `chimes-agent llm-agent --for-machine <profile>` re-picks
  `models.preferred` / `fallback` for the node you will serve on.
- **Tools:** EZRA's `tools:` field is reserved, so tool calls come from the
  harness: Claude Code (`ANTHROPIC_*` against llama-server) or Cline
  (`cline auth --provider openai-compatible ...`), both printed by
  `chimes-agent llm-serve --status`. `ezra serve --agent chimes-assistant`
  applies this prompt for plain chat.
- **Keep in sync:** when `CLAUDE.md` gains a rule that prevents an
  expensive mistake, add it to `prompt.md` too.
