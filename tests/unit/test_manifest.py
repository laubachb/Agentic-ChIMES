"""Regression test: two different stages writing into the same --output-dir
(the normal chaining pattern, e.g. fm-setup-gen then amat-build into one
run directory) must not collide on manifest state, and a stage re-invoked
with the same inputs must short-circuit while different inputs are refused."""

import tempfile
from pathlib import Path

import pytest

from agentic_chimes.stages import _manifest


def test_different_stages_same_output_dir_do_not_collide():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        decision_a, _ = _manifest.begin(d, "stage-a", {"x": 1})
        assert decision_a == "run"
        _manifest.finish(d, "stage-a", {"x": 1}, {"out": "a"})

        # stage-b writing into the SAME directory must not see stage-a's manifest
        decision_b, _ = _manifest.begin(d, "stage-b", {"y": 2})
        assert decision_b == "run"
        _manifest.finish(d, "stage-b", {"y": 2}, {"out": "b"})

        assert _manifest.load(d, "stage-a")["outputs"] == {"out": "a"}
        assert _manifest.load(d, "stage-b")["outputs"] == {"out": "b"}


def test_same_inputs_short_circuits():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _manifest.begin(d, "stage-a", {"x": 1})
        _manifest.finish(d, "stage-a", {"x": 1}, {"out": "a"})

        decision, prior = _manifest.begin(d, "stage-a", {"x": 1})
        assert decision == "short_circuit"
        assert prior["outputs"] == {"out": "a"}


def test_different_inputs_refused_without_force():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _manifest.begin(d, "stage-a", {"x": 1})
        _manifest.finish(d, "stage-a", {"x": 1}, {"out": "a"})

        with pytest.raises(_manifest.InputMismatch):
            _manifest.begin(d, "stage-a", {"x": 2})


def test_different_inputs_allowed_with_force():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _manifest.begin(d, "stage-a", {"x": 1})
        _manifest.finish(d, "stage-a", {"x": 1}, {"out": "a"})

        decision, _ = _manifest.begin(d, "stage-a", {"x": 2}, force=True)
        assert decision == "run"
