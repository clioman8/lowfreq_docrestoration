"""Write a reproducible run plan; never launch long training implicitly."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from .common import read_json, write_json, config_hash


def make_plan(config_path, data_root, tessdata_dir, runs_root, stage="pilot", select_from=None):
    config = read_json(config_path)
    seeds = config["training"].get("seeds", [0, 1, 2])
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Seeds must be a nonempty unique list")
    weights = config["loss"]["lambda_grid"]
    if not weights or len(weights) != len(set(weights)):
        raise ValueError("lambda_grid must be nonempty with distinct values")
    root = Path(runs_root)
    def task(group, seed, weight):
        tag = format(weight, ".8g").replace(".", "p")
        run = root / f"{group}_lambda{tag}_seed{seed}"
        return {"group": group, "seed": seed, "lambda": weight, "run_dir": str(run),
                "argv": [sys.executable, "-m", "docrestore.train", "fit", "--config", str(config_path),
                         "--data-root", str(data_root), "--tessdata-dir", str(tessdata_dir),
                         "--run-dir", str(run), "--group", group, "--seed", str(seed),
                         "--lambda-value", str(weight), "--stage", stage]}
    first = [task("A", seeds[0], 0.0)]
    first += [task(group, seeds[0], weight) for group in ("B", "C") for weight in weights]
    selected, remaining = {"A": 0.0}, []
    if select_from:
        selection_root = Path(select_from)
        for group in ("B", "C"):
            candidates = []
            for job in first:
                if job["group"] != group:
                    continue
                summary_path = selection_root / Path(job["run_dir"]).name / "summary.json"
                summary = read_json(summary_path)
                metadata = read_json(summary_path.parent / "run_metadata.json")
                if (metadata.get("config_sha256") != config_hash(config)
                        or metadata.get("stage") != stage
                        or metadata.get("group") != group
                        or metadata.get("seed") != seeds[0]
                        or metadata.get("lambda") != job["lambda"]):
                    raise ValueError(f"Candidate run metadata does not match this plan: {summary_path}")
                candidates.append((summary["best_validation_max_shifted_cer"],
                                   summary["best_validation_mean_shifted_cer"], job["lambda"]))
            selected[group] = min(candidates)[2]
        remaining = [task(group, seed, selected[group]) for group in ("A", "B", "C") for seed in seeds[1:]]
    return {"stage": stage, "search_seed": seeds[0], "phase1_candidates": first,
            "selected_lambda": selected, "phase2_remaining_seeds": remaining,
            "phase2_status": "resolved" if select_from else "run phase1, then call --select-from",
            "planned_executed_runs_with_seed0_reuse": len(first)+3*(len(seeds)-1),
            "final_reported_runs": 3*len(seeds),
            "selection_rule": "validation maximum shifted corpus CER, then mean shifted CER, then smaller lambda",
            "note": "The chosen seed-0 candidate is reused as the first final repetition. All candidates use the same budget. No test OCR is read by this planner."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data-root", "tessdata-dir", "runs-root", "output"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--stage", choices=["pilot", "final"], default="pilot")
    p.add_argument("--select-from")
    a=p.parse_args()
    plan=make_plan(a.config,a.data_root,a.tessdata_dir,a.runs_root,a.stage,a.select_from)
    write_json(a.output,plan)
    print(f"Wrote {a.output}: {plan['planned_executed_runs_with_seed0_reuse']} planned executions, {plan['final_reported_runs']} reported repetitions")


if __name__ == "__main__":
    main()
