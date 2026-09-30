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


def test_changed_input_file_is_detected(tmp_path):
    data = tmp_path / "train.xyzf"
    data.write_text("frame 1\n")
    out = tmp_path / "run"
    inputs = {"train_xyzf": str(data), "order": 12}
    _manifest.begin(out, "stage-a", inputs)
    _manifest.finish(out, "stage-a", inputs, {"out": "a"})
    assert _manifest.begin(out, "stage-a", inputs)[0] == "short_circuit"

    data.write_text("frame 1\nframe 2 (regenerated in place)\n")
    with pytest.raises(_manifest.InputMismatch, match="train.xyzf"):
        _manifest.begin(out, "stage-a", inputs)
    assert _manifest.begin(out, "stage-a", inputs, force=True)[0] == "run"


def test_legacy_manifest_without_file_fingerprints_still_short_circuits(tmp_path):
    import hashlib
    import json

    inputs = {"x": 1}
    legacy = hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()
    (tmp_path / ".manifest-stage-a.json").write_text(json.dumps(
        {"stage": "stage-a", "input_hash": legacy, "status": "done", "outputs": {"out": "a"}}))
    decision, prior = _manifest.begin(tmp_path, "stage-a", inputs)
    assert decision == "short_circuit" and prior["outputs"] == {"out": "a"}


def test_failed_run_can_be_retried_with_new_inputs(tmp_path):
    _manifest.begin(tmp_path, "stage-a", {"x": 1})
    _manifest.finish(tmp_path, "stage-a", {"x": 1}, {"error": "boom"}, status="failed")
    assert _manifest.begin(tmp_path, "stage-a", {"x": 2})[0] == "run"
