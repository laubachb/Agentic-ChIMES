"""stdout must carry exactly one JSON object no matter how noisy a stage or
the native libraries it loads are (chimes_calculator's C++ banner and
al_driver's prints write straight to fd 1) -- an agent parsing stdout would
otherwise fail. Noise goes to a per-stage log file instead.

Runs the CLI in a real subprocess (with one stage's `run` swapped out) since
the contract is about the process's actual file descriptors, which pytest's
in-process capture doesn't faithfully model."""

import json
import subprocess
import sys
import textwrap

import pytest

ARGV = ["dataset-select", "--frames", "unused.xyzf", "--method", "random"]


def _run_cli(stage_body: str, out_dir):
    body = textwrap.indent(textwrap.dedent(stage_body).strip("\n"), "    ")
    code = (
        "import os, sys\n"
        "from agentic_chimes import cli\n"
        "from agentic_chimes.stages import dataset_select\n"
        "def fake_run(args):\n"
        + body
        + "\n"
        "dataset_select.run = fake_run\n"
        f"sys.exit(cli.main({ARGV + ['--output-dir', str(out_dir)]!r}))\n"
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_noisy_stage_keeps_stdout_pure_json(tmp_path):
    proc = _run_cli(
        """
        os.write(1, b"native-library banner written straight to fd 1\\n")
        print("python-level print")
        return {"ok": True}
        """,
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["ok"] is True
    log = (tmp_path / "dataset-select.log").read_text()
    assert "native-library banner" in log and "python-level print" in log
    assert result["stage_log"].endswith("dataset-select.log")


def test_quiet_stage_leaves_no_log_key(tmp_path):
    proc = _run_cli("return {'ok': True}", tmp_path)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert "stage_log" not in result
    assert not (tmp_path / "dataset-select.log").exists()


def test_failing_stage_reports_error_with_log_tail(tmp_path):
    proc = _run_cli(
        """
        os.write(1, b"last thing the library said before dying\\n")
        raise RuntimeError("boom")
        """,
        tmp_path,
    )
    assert proc.returncode == 1
    result = json.loads(proc.stdout)
    assert result["error"] == "boom"
    assert "last thing the library said" in result["log_tail"]
    assert result["log"].endswith("dataset-select.log")


def test_dry_run_does_not_block_the_real_run(tmp_path):
    """dry-run then real submission with identical inputs: the real call must
    execute, not short-circuit to the dry run's recorded result."""
    body = "return {'dry': bool(getattr(args, 'dry_run', False))}"
    code = (
        "import sys\n"
        "from agentic_chimes import cli\n"
        "from agentic_chimes.stages import dataset_select\n"
        f"dataset_select.run = lambda args: {body.replace('return ', '')}\n"
        "dataset_select.SUPPORTS_DRY_RUN = True  # stand-in for a submitting stage\n"
        "argv = " + repr(ARGV + ["--output-dir", str(tmp_path)]) + "\n"
        "cli.main(argv + ['--dry-run']); cli.main(argv)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    outs = [json.loads(chunk) for chunk in proc.stdout.replace("}\n{", "}\n\x00{").split("\x00")]
    assert outs[0]["dry"] is True and outs[1]["dry"] is False


def test_json_in_rejects_unknown_keys_and_wrong_types(tmp_path):
    import json as _json

    from agentic_chimes import cli as _cli

    bad = tmp_path / "typo.json"
    bad.write_text(_json.dumps({"holdout_xyzff": "h.xyzf"}))
    parser = _cli.build_parser()
    args = parser.parse_args(["evaluate", "--json-in", str(bad)])
    with pytest.raises(ValueError, match="unknown key"):
        _cli._apply_json_in(args, args._module.SCHEMA)

    wrong = tmp_path / "type.json"
    wrong.write_text(_json.dumps({"max_frames": "ten"}))
    args = parser.parse_args(["evaluate", "--json-in", str(wrong)])
    with pytest.raises(ValueError, match="max_frames"):
        _cli._apply_json_in(args, args._module.SCHEMA)


def test_dry_run_refused_by_stages_without_one(capsys):
    from agentic_chimes import cli as _cli

    rc = _cli.main(["dataset-select", "--dry-run", "--frames", "x.xyzf"])
    assert rc == 1
    assert "no --dry-run mode" in capsys.readouterr().out


def test_doctor_quick_reports_structured_checks():
    from types import SimpleNamespace

    from agentic_chimes.stages import doctor

    res = doctor.run(SimpleNamespace(machine=None, quick=True))
    assert {"ok", "n_fail", "n_warn", "checks"} <= set(res)
    assert all(c["status"] in ("ok", "warn", "fail") for c in res["checks"])
    assert all("fix" in c for c in res["checks"] if c["status"] != "ok")
