"""Shared helper for stages that call other stages' `run()` functions
directly (in-process, not by re-invoking the CLI as a subprocess) -- used
by model_build.py and sweep.py to compose fm-setup-gen/amat-build/solve/
evaluate. See docs/concepts/stages_and_contracts.md: "no stage calls
another stage internally" refers to the CLI layer; a *composite* stage
(explicitly a small pipeline over other stages) is exactly what these are,
and they call the underlying `run(args)` functions directly rather than
shelling out to `chimes-agent <stage>`.
"""

from __future__ import annotations

from types import SimpleNamespace


def ns(**kwargs) -> SimpleNamespace:
    """Build a minimal argparse.Namespace-alike for calling another stage's
    run() directly. Every stage's run() only reads specific attributes
    (documented in that stage's module); callers here must supply whatever
    that stage actually accesses, since SimpleNamespace has no defaults."""
    return SimpleNamespace(**kwargs)
