"""Package a study's final model for use: params.txt, fm_setup.in, an
example LAMMPS input, provenance, and MODEL_CARD.md stating what the model
is valid for, how accurate it is, and what it costs to run.

Reads the study registry (`study` stage): params, fm_setup, data_manifest,
hyper_report, benchmark, usage. Missing pieces are reported as gaps in the
card, never invented.
"""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path

from . import study as study_stage
from .lammps_run import _render_input
from ..io import atomic
from ..io import fs

NAME = "deploy"
SUMMARY = "Package the final model (params, LAMMPS example, provenance, MODEL_CARD.md) into the study's 06_deploy/."
USES_OUTPUT_DIR = False
SCHEMA = {
    "type": "object",
    "required": ["study"],
    "properties": {
        "study": {"type": "string"},
        "model_name": {"type": ["string", "null"]},
        "output": {"type": ["string", "null"], "description": "Default <study>/06_deploy."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--study", default=None)
    parser.add_argument("--model-name", dest="model_name", default=None)
    parser.add_argument("--output", default=None)


def _load(path):
    return json.loads(Path(path).read_text()) if path and Path(path).is_file() else None


def model_facts(root: Path) -> dict:
    """Everything the card and the report state about the final model."""
    get = lambda k: study_stage.artifact(root, k)  # noqa: E731
    manifest, hyper = _load(get("data_manifest")), _load(get("hyper_report"))
    bench = _load(get("benchmark"))
    if bench is None and get("benchmark") and Path(get("benchmark")).is_dir():
        bench = _load(Path(get("benchmark")) / "benchmark.json")
    usage = _load(get("usage"))
    facts = {"study": str(root), "params": get("params"), "fm_setup": get("fm_setup"), "gaps": []}
    if manifest:
        facts["data"] = {k: manifest.get(k) for k in ("elements", "level_of_theory", "labeled", "units")}
        facts["data"]["summary"] = manifest.get("summary")
        facts["data"]["pairs"] = manifest.get("pairs")
        facts["data"]["sources"] = manifest.get("sources")
        facts["data"]["warnings"] = manifest.get("warnings")
    else:
        facts["gaps"].append("no data_manifest registered")
    if hyper:
        facts["hyperparameters"] = hyper.get("hyperparameters")
        facts["accuracy"] = {k: hyper["final"].get(k) for k in ("holdout_relative_force_error", "holdout_relative_force_se",
                                                                 "holdout_rmse_force", "holdout_rmse_energy_per_atom",
                                                                 "train_relative_force_error", "n_params", "nlayers")}
        facts["fit_notes"] = hyper.get("notes")
    else:
        facts["gaps"].append("no hyper_report registered")
    if bench:
        facts["cost"] = {"cost_model": bench.get("cost_model"), "estimates_1ns": bench.get("estimates_1ns"),
                         "strong_recommended": (bench.get("results", {}).get("strong") or {}).get("recommended_ranks"),
                         "weak_recommended": (bench.get("results", {}).get("weak") or {}).get("recommended_ranks"),
                         "machine": bench.get("machine"), "unstable_cases": bench.get("unstable"),
                         "notes": bench.get("notes")}
    else:
        facts["gaps"].append("no benchmark: runtime cost unknown")
    if usage:
        facts["development_cost"] = {k: usage.get(k) for k in ("total_cpu_hours", "total_used_cpu_hours",
                                                               "allocation_efficiency", "total_local_cpu_hours", "by_phase")}
    else:
        facts["gaps"].append("no usage report: development cost unknown")
    return facts


def _fmt(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}g}" if isinstance(x, float) else str(x))


def model_card(name: str, f: dict) -> str:
    L = [f"# Model card: {name}", "",
         f"Generated {datetime.date.today().isoformat()} by `chimes-agent deploy` from study `{f['study']}`.", ""]
    hp, acc, data = f.get("hyperparameters") or {}, f.get("accuracy") or {}, f.get("data") or {}
    L += ["## What it is", "",
          f"ChIMES many-body potential for **{'-'.join(hp.get('elements') or data.get('elements') or [])}**, "
          f"fitted to {'/'.join(data.get('level_of_theory') or ['?'])} labels.", ""]
    if hp:
        order = hp.get("order", {})
        L += ["| setting | value |", "|---|---|",
              f"| polynomial orders (2b/3b/4b) | {order.get('2')}/{order.get('3') or 0}/{order.get('4') or 0} |",
              f"| outer cutoffs 2b / 3b / 4b (Å) | {next(iter(hp['pair_cutoffs'].values()))[1]} / {_fmt(hp.get('special_maxim_3b'))} / {_fmt(hp.get('special_maxim_4b'))} |",
              "| inner cutoffs (Å) | " + ", ".join(f"{p} {v[0]}" for p, v in hp["pair_cutoffs"].items()) + " |",
              "| Morse λ (Å) | " + ", ".join(f"{p} {v}" for p, v in hp["morse_lambda"].items()) + " |",
              f"| excluded 3-body types | {', '.join(' '.join(t) for t in hp.get('exclude_3b') or []) or 'none'} |",
              f"| excluded 4-body types | {', '.join(' '.join(t) for t in hp.get('exclude_4b') or []) or 'none'} |",
              f"| coefficients | {acc.get('n_params')} |", ""]
    L += ["## Accuracy (holdout)", ""]
    if acc:
        L += [f"- force RMSE {_fmt(acc.get('holdout_rmse_force'))} kcal/mol/Å; relative to the reference forces "
              f"{_fmt(acc.get('holdout_relative_force_error'))} ± {_fmt(acc.get('holdout_relative_force_se'), 2)} "
              f"(training {_fmt(acc.get('train_relative_force_error'))})",
              f"- energy RMSE {_fmt(acc.get('holdout_rmse_energy_per_atom'))} kcal/mol/atom", ""]
    else:
        L += ["Not available.", ""]
    L += ["## Where it is valid", ""]
    pairs = data.get("pairs") or {}
    if pairs:
        L += ["Do not run configurations with pair distances below the sampled minimum (the inner cutoff); "
              "ChIMES applies only a penalty there, not physics.", "",
              "| pair | min sampled distance (Å) | frames |", "|---|---|---|"]
        L += [f"| {p} | {_fmt(v.get('min_distance'))} | {v.get('n_frames')} |" for p, v in pairs.items()]
        comp = (data.get("summary") or {}).get("compositions")
        if comp:
            L += ["", "Compositions in the training data: " + ", ".join(f"{k} ({v} frames)" for k, v in comp.items()) + "."]
        L += [""]
    L += ["## Running it (LAMMPS)", "",
          "```", "units real", "atom_style atomic", "pair_style chimesFF", "pair_coeff * * params.txt", "```", "",
          "`in.lammps.example` in this directory is a complete NVT input; types must be ordered as the elements above. "
          "The LAMMPS binary must be built with the ChIMES pair style (`chimes-agent setup --component lammps`).", ""]
    cost = f.get("cost")
    L += ["## Cost to run", ""]
    if cost and cost.get("cost_model"):
        cm = cost["cost_model"]
        L += [f"Measured on {cost.get('machine') or 'the benchmark machine'}: "
              f"{cm['core_s_per_atom_step']:.3g} core-seconds per atom per step ({cm['from']}). "
              f"Strong scaling stays efficient up to {cost.get('strong_recommended')} ranks for the benchmark size.", "",
              "| atoms | 1 ns at 1 fs: CPU-hours | ranks | nodes | wall hours |", "|---|---|---|---|---|"]
        L += [f"| {e['atoms']:,} | {e['cpu_hours']:,} | {e['suggested_ranks']} | {e['suggested_nodes']} | {e['wall_hours_at_suggested']} |"
              for e in cost.get("estimates_1ns") or []]
        L += ["", "CPU-hours = core_s_per_atom_step × atoms × steps / 3600 at the packing the run uses "
              "(per-core cost rises as a node fills); request that plus ~20 % margin."]
        L += [f"- {n}" for n in cost.get("notes") or []] + [""]
    else:
        L += ["Not benchmarked.", ""]
    dev = f.get("development_cost")
    if dev:
        L += ["## Development cost", "", f"{_fmt(dev.get('total_cpu_hours'))} CPU-hours charged "
              f"({_fmt(dev.get('total_used_cpu_hours'))} used; allocation efficiency {_fmt(dev.get('allocation_efficiency'))}), "
              f"plus {_fmt(dev.get('total_local_cpu_hours'))} login-node CPU-hours.", ""]
    sources = data.get("sources") or []
    if sources:
        L += ["## Training data provenance", ""]
        L += [f"- {s.get('source')}: license {s.get('license') or 'unknown'}" + (f", DOI {s['doi']}" if s.get("doi") else "")
              for s in sources]
        L += ["", "Cite the datasets above when using this model.", ""]
    caveats = (f.get("fit_notes") or []) + (data.get("warnings") or [])
    if caveats or f.get("gaps"):
        L += ["## Caveats", ""] + [f"- {c}" for c in caveats] + [f"- (gap) {g}" for g in f.get("gaps", [])] + [""]
    return "\n".join(L)


def run(args) -> dict:
    root = study_stage.find_study(getattr(args, "study", None) or ".")
    if root is None:
        raise ValueError("deploy needs a study (study.json)")
    facts = model_facts(root)
    if not facts.get("params") or not Path(facts["params"]).is_file():
        raise ValueError("no final params.txt registered (study --register params=...)")
    out = Path(getattr(args, "output", None) or root / "06_deploy")
    fs.ensure_dir(out)
    shutil.copy(facts["params"], out / "params.txt")
    if facts.get("fm_setup") and Path(facts["fm_setup"]).is_file():
        shutil.copy(facts["fm_setup"], out / "fm_setup.in")
    (out / "in.lammps.example").write_text(_render_input("md", "structure.data", "params.txt", temperature=300.0,
                                                         nsteps=10000, timestep=1.0))
    name = getattr(args, "model_name", None) or study_stage.load(root).get("name") or root.name
    (out / "MODEL_CARD.md").write_text(model_card(name, facts))
    atomic.write_json((out / "model_facts.json"), facts, indent=1, default=str)
    study_stage.run(type("A", (), {"init": None, "study": str(root), "name": None, "goal": None, "elements": None,
                                   "register": [f"deploy={out}"], "extra_roots": None})())
    return {"deploy_dir": str(out), "files": sorted(p.name for p in out.iterdir()), "gaps": facts["gaps"]}
