#!/usr/bin/env python3
"""PreToolUse hook: ask the user before any chimes-agent call that spends
allocation or starts long-lived work.

settings.json "ask" rules match command text only, so they miss the module
form (`python3 -m agentic_chimes.cli ...`) and a `--machine` hidden inside a
`--json-in` file. This hook parses the command (and reads the JSON file) and
returns "ask" for:

  - submit, al-run (except --status-of), auto-build, qe-relabel (except --collect)
  - any stage with a machine set (flag or JSON) that is not a dry run
  - benchmark --collect is local; hyper-search/solve/amat-build/model-build/
    sweep/benchmark with a machine are submissions
  - raw sbatch

Anything it cannot parse falls through to the normal permission rules.
"""
import json
import shlex
import sys

ALWAYS_ASK = {"submit", "auto-build"}


def _segments(command: str):
    lex = shlex.shlex(command, posix=True, punctuation_chars=";&|")
    lex.whitespace_split = True
    seg = []
    for tok in lex:
        if tok and set(tok) <= set(";&|"):
            if seg:
                yield seg
            seg = []
        else:
            seg.append(tok)
    if seg:
        yield seg


def _stage_args(tokens):
    for i, tok in enumerate(tokens):
        if tok.endswith("chimes-agent"):
            return tokens[i + 1:]
        if tok == "agentic_chimes.cli" and i > 0 and tokens[i - 1] == "-m":
            return tokens[i + 1:]
    return None


def _flag(args, name):
    for i, a in enumerate(args):
        if a == name:
            return args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else True
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def reason_to_ask(command: str):
    for tokens in _segments(command):
        if tokens and tokens[0] == "sbatch":
            return "raw sbatch submission"
        args = _stage_args(tokens)
        if not args:
            continue
        stage = args[0]
        rest = args[1:]
        if "--describe" in rest:
            continue
        data = {}
        json_in = _flag(rest, "--json-in")
        if isinstance(json_in, str):
            try:
                with open(json_in) as f:
                    data = json.load(f) or {}
            except (OSError, ValueError):
                data = {}
        dry = "--dry-run" in rest or bool(data.get("dry_run"))
        machine = _flag(rest, "--machine") or data.get("machine")
        if stage in ALWAYS_ASK and not dry:
            return f"{stage} submits HPC work or runs a long pipeline"
        if stage == "qe-relabel" and not dry and not (_flag(rest, "--collect") or data.get("collect")):
            return "qe-relabel submits QE labeling jobs"
        if stage == "al-run" and not (_flag(rest, "--status-of") or data.get("status_of")):
            return "al-run starts or stops a long-lived al_driver campaign"
        if machine and not dry and not (stage == "benchmark" and (_flag(rest, "--collect") or data.get("collect"))):
            src = "in " + json_in if not _flag(rest, "--machine") else "via --machine"
            return f"{stage} with a machine set ({src}) submits a Slurm job"
    return None


def main():
    try:
        event = json.load(sys.stdin)
    except ValueError:
        return 0
    if event.get("tool_name") != "Bash":
        return 0
    try:
        why = reason_to_ask(event.get("tool_input", {}).get("command", ""))
    except ValueError:  # unbalanced quotes etc.: leave it to the normal rules
        return 0
    if why:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": f"Needs the user's approval: {why}. Show the dry run first.",
        }}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
