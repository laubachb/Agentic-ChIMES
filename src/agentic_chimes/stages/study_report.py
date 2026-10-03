"""Collate a whole study into REPORT_FACTS.json and a REPORT.md skeleton.

Every number in REPORT.md comes from an artifact the phases wrote (data
manifest and curation report, hyper_report, MD logs, benchmark, usage,
deployed model card). Narrative sections are left as marked placeholders
(`<!-- NARRATIVE: ... -->`) for the report agent (chimes-report-writer),
which writes them from the facts. The facts file is the ground truth the
narrative must agree with.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

from . import study as study_stage
from .deploy import _fmt, model_facts
from ..io import atomic

NAME = "study-report"
SUMMARY = "Collate a study's artifacts (data -> fit -> MD -> benchmark -> deploy, CPU-hours) into REPORT_FACTS.json + REPORT.md."
USES_OUTPUT_DIR = False
SCHEMA = {"type": "object", "required": ["study"], "properties": {"study": {"type": "string"}}}


def add_arguments(parser) -> None:
    parser.add_argument("--study", default=None)


def _load(p):
    return json.loads(Path(p).read_text()) if p and Path(p).is_file() else None


def md_run_summary(run_dir: str) -> dict:
    """Stability summary of a LAMMPS run from its log's thermo output."""
    log = Path(run_dir) / "log.lammps"
    if not log.is_file():
        return {"dir": run_dir, "status": "no log"}
    text = log.read_text(errors="replace")
    rows, header = [], None
    for line in text.splitlines():
        toks = line.split()
        if toks and toks[0] == "Step":
            header = toks
            continue
        if header and toks and len(toks) == len(header) and re.fullmatch(r"-?\d+", toks[0]):
            try:
                rows.append(dict(zip(header, map(float, toks))))
            except ValueError:
                pass
        elif header and toks and toks[0] == "Loop":
            header = None
    atoms = re.findall(r"with (\d+) atoms", text)
    natoms = int(atoms[-1]) if atoms else None
    ts = re.findall(r"^\s*timestep\s+([\d.]+)", text, re.M)
    fixes = re.findall(r"^\s*fix\s+\S+\s+\S+\s+(nve|nvt|npt|nph|langevin)", text, re.M)
    ensemble = "NVE" if fixes == ["nve"] else (fixes[0].upper() if fixes else None)
    out = {"dir": run_dir, "atoms": natoms, "n_thermo": len(rows), "ensemble": ensemble,
           "unstable": bool(re.search(r"\bnan\b|Lost atoms|ERROR", text, re.IGNORECASE))}
    if rows:
        steps = rows[-1]["Step"] - rows[0]["Step"]
        out["steps"] = int(steps)
        if ts:
            out["simulated_ps"] = round(steps * float(ts[-1]) / 1000.0, 3)
        if "Temp" in rows[0]:
            temps = [r["Temp"] for r in rows[len(rows) // 5:]]
            out["mean_temp"] = round(sum(temps) / len(temps), 1)
        e_key = "TotEng" if "TotEng" in rows[0] else None
        if e_key and natoms and steps:
            change = round((rows[-1][e_key] - rows[0][e_key]) / natoms, 4)
            # only an NVE run conserves energy; under a thermostat the change
            # includes heat exchanged with the bath and relaxation of the start
            out["energy_drift_kcal_mol_atom" if ensemble == "NVE" else "total_energy_change_kcal_mol_atom"] = change
    loop = re.findall(r"Loop time of ([\d.]+) on (\d+) procs", text)
    if loop:
        out["loop_time_s"], out["procs"] = float(loop[-1][0]), int(loop[-1][1])
    return out


def collect(root: Path) -> dict:
    s = study_stage.load(root)
    f = model_facts(root)
    manifest_path = study_stage.artifact(root, "data_manifest")
    manifest = _load(manifest_path)
    curation = _load(manifest.get("curation_report")) if manifest else None
    hyper = _load(study_stage.artifact(root, "hyper_report"))
    md = [md_run_summary(d) for d in (study_stage.artifact(root, "md_runs") or [])]
    al_dir = study_stage.artifact(root, "al_run")
    al = None
    if al_dir and Path(al_dir).is_dir():
        alcs = sorted(p.name for p in Path(al_dir).glob("ALC-*") if p.is_dir())
        al = {"dir": al_dir, "cycles": alcs, "driver_log": str(Path(al_dir) / "driver.log")}
    texts = {}
    for name, p in (("study_md", root / "STUDY.md"), ("data_plan", root / "01_data" / "DATA_PLAN.md"),
                    ("hyper_report_md", root / "02_fit" / "HYPER_REPORT.md")):
        if p.is_file():
            texts[name] = p.read_text()[:20000]
    return {
        "study": {k: s.get(k) for k in ("name", "goal", "elements", "created_at")},
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "data": {"manifest": manifest_path, "summary": (manifest or {}).get("summary"),
                 "level_of_theory": (manifest or {}).get("level_of_theory"), "sources": (manifest or {}).get("sources"),
                 "pairs": (manifest or {}).get("pairs"), "warnings": (manifest or {}).get("warnings"),
                 "n_train": None, "curation": {k: (curation or {}).get(k) for k in ("n_input", "n_removed", "selection")},
                 "removed_by_reason": {}},
        "fit": {"stages": [{k: st.get(k) for k in ("stage", "n_points", "reason", "skipped")} for st in (hyper or {}).get("stages", [])],
                "exclusion_analysis": next((st.get("cluster_types") for st in (hyper or {}).get("stages", []) if st.get("stage") == "exclude"), None),
                "four_body": next((st.get("four_body_gain_by_solver") for st in (hyper or {}).get("stages", []) if st.get("stage") == "4b"), None),
                "n_fits": (hyper or {}).get("n_fits"), "notes": (hyper or {}).get("notes")},
        "model": {k: f.get(k) for k in ("hyperparameters", "accuracy", "params")},
        "active_learning": al,
        "md_runs": md,
        "benchmark": f.get("cost"),
        "usage": f.get("development_cost"),
        "deploy": study_stage.artifact(root, "deploy"),
        "gaps": f.get("gaps", []) + ([] if md else ["no MD runs registered (04_md)"]) + ([] if al else ["no active-learning run"]),
        "texts": texts,
    }


def _curation_counts(facts, root):
    mp = facts["data"]["manifest"]
    m = _load(mp)
    if not m:
        return
    c = _load(m.get("curation_report"))
    if c:
        facts["data"]["removed_by_reason"] = {}
        for r in c.get("removed", []):
            key = r["reason"].split(":")[0].split()[0]
            facts["data"]["removed_by_reason"][key] = facts["data"]["removed_by_reason"].get(key, 0) + 1
    facts["data"]["n_train"] = m.get("n_train") or (m.get("summary") or {}).get("n_frames")


def render(facts: dict) -> str:
    st, d, fit, mdl = facts["study"], facts["data"], facts["fit"], facts["model"]
    acc, hp = mdl.get("accuracy") or {}, mdl.get("hyperparameters") or {}
    N = lambda what: f"<!-- NARRATIVE: {what} -->"  # noqa: E731
    L = [f"# ChIMES model development report: {st.get('name')}", "",
         f"*Study `{facts.get('deploy') and Path(facts['deploy']).parent or ''}`; generated {facts['generated_at']}; "
         "numbers from `REPORT_FACTS.json`.*", "",
         "## Summary", "", N("3-5 sentences: what was built, for what, how good, how expensive, main caveat"), "",
         f"**Goal:** {st.get('goal') or 'n/a'}", ""]
    if acc:
        L += ["| | |", "|---|---|",
              f"| elements | {'-'.join(st.get('elements') or hp.get('elements') or [])} |",
              f"| level of theory | {', '.join(d.get('level_of_theory') or ['n/a'])} |",
              f"| training frames | {d.get('n_train') or 'n/a'} |",
              f"| holdout force error (relative) | {_fmt(acc.get('holdout_relative_force_error'))} ± {_fmt(acc.get('holdout_relative_force_se'), 2)} |",
              f"| holdout energy RMSE | {_fmt(acc.get('holdout_rmse_energy_per_atom'))} kcal/mol/atom |",
              f"| coefficients | {acc.get('n_params')} |"]
        if facts.get("usage"):
            L += [f"| development cost | {_fmt(facts['usage'].get('total_cpu_hours'))} CPU-hours charged |"]
        if facts.get("benchmark") and facts["benchmark"].get("cost_model"):
            L += [f"| runtime cost | {facts['benchmark']['cost_model']['core_s_per_atom_step']:.3g} core-s/atom-step |"]
        L += [""]
    L += ["## 1. Data", "", N("where the data came from, why these sources, what curation removed and why, coverage gaps"), ""]
    summ = d.get("summary") or {}
    if summ:
        L += [f"- compositions: " + ", ".join(f"{k} ({v})" for k, v in (summ.get("compositions") or {}).items()),
              f"- atoms per frame: {summ.get('natoms_range')}; thinnest cell {_fmt(summ.get('thinnest_cell_width_ang'))} Å",
              f"- removed in curation: " + (", ".join(f"{k} {v}" for k, v in d.get("removed_by_reason", {}).items()) or "none")]
        for s in d.get("sources") or []:
            L += [f"- source: {s.get('source')} (license {s.get('license') or 'unknown'})"]
        L += [""]
        if d.get("pairs"):
            L += ["| pair | frames | min distance (Å) | short-range samples |", "|---|---|---|---|"]
            L += [f"| {p} | {v.get('n_frames')} | {_fmt(v.get('min_distance'))} | {v.get('n_within_1.2x_min')} |" for p, v in d["pairs"].items()]
            L += [""]
    L += ["## 2. Hyperparameters", "", N("how cutoffs, lambdas and orders were chosen; what each stage decided and why; what 4-body and exclusions showed"), ""]
    if fit.get("stages"):
        L += ["| stage | fits | decision |", "|---|---|---|"]
        L += [f"| {s['stage']} | {s.get('n_points') or ('skipped' if s.get('skipped') else '')} | {s.get('reason') or ''} |" for s in fit["stages"]]
        L += [""]
    if fit.get("exclusion_analysis"):
        L += ["Cluster-type exclusion analysis (score change when removed; positive = needed):", "",
              "| type | instances | coefficients saved | score change | excluded |", "|---|---|---|---|---|"]
        for c in fit["exclusion_analysis"]:
            ch = c.get("score_change_when_removed") or {}
            L += [f"| {c['type']} | {c.get('instances')} | {c.get('coefficients_saved')} | {', '.join(f'{v:+.3f}' for v in ch.values()) or 'n/a'} | {'yes' if c.get('excluded') else 'no'} |"]
        L += [""]
    L += ["## 3. Final model", "", N("the chosen model in words; accuracy in context; what the training/holdout gap says"), ""]
    if hp:
        L += ["```json", json.dumps({k: hp.get(k) for k in ("order", "pair_cutoffs", "morse_lambda", "special_maxim_3b",
                                                              "special_maxim_4b", "exclude_3b", "exclude_4b", "nlayers",
                                                              "algorithm", "alpha")}, indent=1), "```", ""]
    L += ["## 4. Active learning", ""]
    L += ([f"Cycles: {', '.join(facts['active_learning']['cycles'])} (`{facts['active_learning']['dir']}`).", "",
           N("what active learning changed")] if facts.get("active_learning") else ["Not run in this study.", ""])
    L += ["", "## 5. MD validation", ""]
    if facts["md_runs"]:
        L += ["| run | ensemble | atoms | simulated ps | mean T (K) | energy change (kcal/mol/atom) | stable |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {Path(m['dir']).name} | {m.get('ensemble')} | {m.get('atoms')} | {m.get('simulated_ps')} | {m.get('mean_temp')} | "
              f"{m.get('energy_drift_kcal_mol_atom', m.get('total_energy_change_kcal_mol_atom'))} | {'NO' if m.get('unstable') else 'yes'} |"
              for m in facts["md_runs"]]
        L += ["", "Energy change is drift only for NVE runs; under a thermostat it includes heat exchanged with the bath."]
        L += ["", N("what the MD runs show about stability and physical behavior")]
    else:
        L += ["No MD runs registered."]
    L += ["", "## 6. Performance and compute requests", "", N("how the model scales, recommended ranks, and how to size a request")]
    b = facts.get("benchmark")
    if b and b.get("cost_model"):
        L += ["", f"Cost unit: {b['cost_model']['core_s_per_atom_step']:.3g} core-seconds per atom-step ({b['cost_model']['from']}); "
              f"strong scaling efficient to {b.get('strong_recommended')} ranks, weak scaling to {b.get('weak_recommended')}.", "",
              "| atoms | 1 ns (1 fs steps): CPU-hours | ranks | nodes | wall hours |", "|---|---|---|---|---|"]
        L += [f"| {e['atoms']:,} | {e['cpu_hours']:,} | {e['suggested_ranks']} | {e['suggested_nodes']} | {e['wall_hours_at_suggested']} |"
              for e in b.get("estimates_1ns") or []]
    L += ["", "## 7. Development cost", ""]
    u = facts.get("usage")
    if u:
        L += [f"{_fmt(u.get('total_cpu_hours'))} CPU-hours charged, {_fmt(u.get('total_used_cpu_hours'))} used "
              f"(allocation efficiency {_fmt(u.get('allocation_efficiency'))}); {_fmt(u.get('total_local_cpu_hours'))} login-node CPU-hours.", "",
              "| phase | jobs | CPU-hours charged | used |", "|---|---|---|---|"]
        L += [f"| {p} | {v.get('jobs')} | {_fmt(v.get('cpu_hours'))} | {_fmt(v.get('used_cpu_hours'))} |" for p, v in (u.get("by_phase") or {}).items()]
        L += ["", N("where the compute went and what would make the next study cheaper")]
    else:
        L += ["Not measured."]
    L += ["", "## 8. Findings, caveats and next steps", "", N("numbered findings; caveats (including every gap below); concrete next steps"), ""]
    if facts.get("gaps"):
        L += ["Gaps in this study:", ""] + [f"- {g}" for g in facts["gaps"]] + [""]
    if fit.get("notes"):
        L += ["Search notes:", ""] + [f"- {n}" for n in fit["notes"]] + [""]
    if facts.get("deploy"):
        L += [f"Deployed model and model card: `{facts['deploy']}`.", ""]
    return "\n".join(L)


def run(args) -> dict:
    root = study_stage.find_study(getattr(args, "study", None) or ".")
    if root is None:
        raise ValueError("study-report needs a study (study.json)")
    facts = collect(root)
    _curation_counts(facts, root)
    atomic.write_json((root / "REPORT_FACTS.json"), facts, indent=1, default=str)
    report = root / "REPORT.md"
    text = render(facts)
    (root / "REPORT.md").write_text(text)
    return {"report": str(report), "facts": str(root / "REPORT_FACTS.json"),
            "narrative_placeholders": text.count("<!-- NARRATIVE:"), "gaps": facts["gaps"]}
