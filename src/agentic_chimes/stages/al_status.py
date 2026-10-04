"""Where an active-learning campaign stands, across rounds, with a verdict.

Layout (the chimes-active-learning playbook):

    <study>/03_al/round<k>/
        batch/batch.json          al-batch: what was chosen and why
        qe/                       qe-relabel work dir (labeled.xyzf, provenance.json)
        merge/data_manifest.json  al-merge: training set after the round
        fit/params.txt            refit at the chosen hyperparameters
        md/run/md_check.json      md-check of the refit (optional)
        quests/quests.json        quests coverage of the round's harvest (optional)

For every round with a `fit/params.txt`, the model is scored on the study's
fixed holdout (the one in the base data manifest), so errors are comparable
across rounds. The verdict follows the playbook's stopping rule:

    CONVERGED   the latest round is stable at every checked temperature,
                samples nothing inside the inner cutoffs, and its harvest is
                no longer novel (QUESTS fraction of novel frames < 0.1, or
                fingerprint not distinguishable) -- or the holdout error has
                not improved beyond its noise for two rounds while MD is stable
    CONTINUE    otherwise, with the reasons listed
    NO_ROUNDS   nothing to assess yet
"""

from __future__ import annotations

import re
from pathlib import Path

from ..io import atomic, fs
from ..io import xyzf as xyzf_io

NAME = "al-status"
SUMMARY = "Active-learning campaign status: per-round holdout errors, stability, novelty, and a CONVERGED/CONTINUE verdict."
USES_OUTPUT_DIR = False
SCHEMA = {
    "type": "object",
    "properties": {
        "study": {"type": ["string", "null"], "description": "Study directory (rounds under 03_al/round<k>)."},
        "al_dir": {"type": ["string", "null"], "description": "Directory holding round<k>/ (default <study>/03_al)."},
        "base_manifest": {"type": ["string", "null"], "description": "data_manifest.json with the fixed holdout (default: the study's registered one)."},
        "base_params": {"type": ["string", "null"], "description": "Model before active learning (round 0)."},
        "plot": {"type": "boolean", "default": True},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--study", default=None)
    parser.add_argument("--al-dir", dest="al_dir", default=None)
    parser.add_argument("--base-manifest", dest="base_manifest", default=None)
    parser.add_argument("--base-params", dest="base_params", default=None)
    parser.add_argument("--no-plot", dest="plot", action="store_false", default=True)


def _round_index(p: Path):
    m = re.fullmatch(r"round(\d+)", p.name)
    return int(m.group(1)) if m else None


def run(args) -> dict:
    from . import evaluate, study as study_stage

    root = study_stage.find_study(getattr(args, "study", None) or ".") if getattr(args, "study", None) or not getattr(args, "al_dir", None) else None
    al_dir = Path(getattr(args, "al_dir", None) or (root / "03_al" if root else "03_al")).resolve()
    manifest_path = getattr(args, "base_manifest", None) or (study_stage.artifact(root, "data_manifest") if root else None)
    manifest = atomic.read_json(manifest_path) if manifest_path else None
    holdout = xyzf_io.read_xyzf(manifest["holdout_xyzf"]) if manifest and manifest.get("holdout_xyzf") else None
    base_params = getattr(args, "base_params", None) or (study_stage.artifact(root, "params") if root else None)

    rounds = sorted((p for p in al_dir.glob("round*") if p.is_dir() and _round_index(p) is not None), key=_round_index)
    rows = []
    if base_params and Path(base_params).is_file() and holdout:
        ev = evaluate.evaluate_frames(holdout, [str(base_params)])["results"][0]
        rows.append({"round": 0, "n_train": manifest.get("n_train"), "n_added": 0, "params": str(base_params),
                     "holdout_relative_force_error": round(ev["relative_force_error"], 4),
                     "holdout_rmse_energy_per_atom": ev["rmse_energy_kcal_mol_per_atom"],
                     "by_composition": {g: v["relative_force_error"] for g, v in ev["by_composition"].items()}})
    for rd in rounds:
        k = _round_index(rd)
        row = {"round": k}
        batch = atomic.read_json(rd / "batch" / "batch.json")
        if batch:
            row.update({"batch_size": batch.get("n_selected"), "batch_close_contact": batch.get("n_selected_close_contact"),
                        "batch_signals": batch.get("signals_used")})
        merged = atomic.read_json(rd / "merge" / "data_manifest.json")
        if merged:
            last = (merged.get("al_rounds") or [{}])[-1]
            row.update({"n_train": merged.get("n_train"), "n_added": last.get("n_added"),
                        "n_duplicates_dropped": last.get("n_duplicates_dropped")})
        params = rd / "fit" / "params.txt"
        if params.is_file() and holdout:
            ev = evaluate.evaluate_frames(holdout, [str(params)])["results"][0]
            row.update({"params": str(params), "holdout_relative_force_error": round(ev["relative_force_error"], 4),
                        "holdout_rmse_energy_per_atom": ev["rmse_energy_kcal_mol_per_atom"],
                        "by_composition": {g: v["relative_force_error"] for g, v in ev["by_composition"].items()}})
        md = atomic.read_json(rd / "md" / "run" / "md_check.json") or atomic.read_json(rd / "md" / "md_check.json")
        if md and md.get("models"):
            m = md["models"][0]
            row.update({"md_stable": m.get("stable_at_all_temperatures"), "md_unstable_T": m.get("unstable_temperatures"),
                        "md_below_inner_cutoff": m.get("below_inner_cutoff_frames"),
                        "md_close_contact_fraction": m.get("max_close_contact_fraction")})
        q = atomic.read_json(rd / "quests" / "quests.json")
        if q:
            row.update({"novel_fraction_frames": q.get("fraction_novel_frames"), "entropy_gain": q.get("entropy_gain_if_all_added")})
        fpj = atomic.read_json(rd / "fingerprint" / "fingerprint.json") or atomic.read_json(rd / "fingerprint" / "run" / "fingerprint.json")
        if fpj and fpj.get("sets"):
            row["fingerprint_distinguishable"] = fpj["sets"].get("distinguishable")
        rows.append(row)

    if not rounds:
        verdict, reasons = "NO_ROUNDS", ["no round<k> directories under " + str(al_dir)]
    else:
        last = rows[-1]
        reasons = []
        if last.get("md_stable") is False:
            reasons.append(f"MD unstable at {last.get('md_unstable_T')}")
        if last.get("md_below_inner_cutoff"):
            reasons.append(f"{last['md_below_inner_cutoff']} MD frames inside an inner cutoff")
        if last.get("novel_fraction_frames") is not None and last["novel_fraction_frames"] >= 0.1:
            reasons.append(f"{last['novel_fraction_frames']:.0%} of the harvest is novel (QUESTS)")
        if last.get("fingerprint_distinguishable"):
            reasons.append("harvest distinguishable from training (fingerprint D2)")
        errs = [r["holdout_relative_force_error"] for r in rows if r.get("holdout_relative_force_error") is not None]
        stalled = len(errs) >= 3 and all(abs(errs[-1] - e) < 0.02 for e in errs[-3:-1])
        if last.get("md_stable") is None and not errs:
            reasons.append("no fit or md-check found for the latest round")
        checked = last.get("md_stable") is not None
        if not reasons and checked and (last.get("novel_fraction_frames") is not None or last.get("fingerprint_distinguishable") is not None):
            verdict = "CONVERGED"
        elif stalled and last.get("md_stable") is True and not last.get("md_below_inner_cutoff"):
            verdict, reasons = "CONVERGED", [f"holdout error flat for 3 rounds ({', '.join(f'{e:.3f}' for e in errs[-3:])}) with stable MD"]
        else:
            verdict = "CONTINUE"
            if not reasons:
                reasons.append("stopping evidence incomplete: run md-check and quests/fingerprint on the latest round's harvest")
    result = {"al_dir": str(al_dir), "rounds": rows, "verdict": verdict, "reasons": reasons}
    fs.ensure_dir(al_dir)
    if getattr(args, "plot", True):
        from ..io import plots

        pts = [(r["round"], r["holdout_relative_force_error"]) for r in rows if r.get("holdout_relative_force_error") is not None]
        if len(pts) > 1:
            png = plots.lines({"holdout": ([p[0] for p in pts], [p[1] for p in pts])}, al_dir / "al_progress.png",
                              xlabel="active-learning round", ylabel="holdout relative force error", title="Active learning progress")
            if png:
                result["plot"] = png
    atomic.write_json(al_dir / "AL_STATUS.json", result, indent=1)
    L = ["# Active-learning status", "", f"Verdict: **{verdict}**", ""] + [f"- {r}" for r in reasons] + ["",
         "| round | train frames | added | holdout F err | E err | MD stable | inside inner cutoff | novel frames |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['round']} | {r.get('n_train', '')} | {r.get('n_added', '')} | {r.get('holdout_relative_force_error', '')} | "
                 f"{'' if r.get('holdout_rmse_energy_per_atom') is None else round(r['holdout_rmse_energy_per_atom'], 3)} | "
                 f"{r.get('md_stable', '')} | {r.get('md_below_inner_cutoff', '')} | {r.get('novel_fraction_frames', '')} |")
    (al_dir / "AL_STATUS.md").write_text("\n".join(L) + "\n")
    return result
