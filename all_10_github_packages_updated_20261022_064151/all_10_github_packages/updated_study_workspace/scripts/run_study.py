"""Run the frozen study sequentially with resumable, provenance checked stages.

Launch only after configs/frozen.json and provenance/study_lock.json exist.
All scientific choices are frozen before this script renders final test images.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "experiment"
PYTHON = ROOT / ".venv/Scripts/python.exe"
CONFIG = PROJECT / "configs/frozen.json"
STATE = PROJECT / "study_status.json"
LOGS = PROJECT / "logs"
DATA = PROJECT / "data/main"
RUNS = PROJECT / "runs/main"
TESS = PROJECT / "tools/tesseract/tesseract.exe"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_lock():
    lock = read(PROJECT / "provenance/study_lock.json")
    if digest(CONFIG) != lock["config_sha256"]:
        raise ValueError("Frozen configuration has changed")
    for relative, expected in lock["source_sha256"].items():
        if digest(PROJECT / relative) != expected:
            raise ValueError(f"Frozen scientific source changed: {relative}")


def command(name, argv, expected=None, resume_training=False):
    check_lock()
    state = read(STATE)
    if name in state["completed"]:
        if expected and not Path(expected).is_file():
            raise FileNotFoundError(f"Completed stage lost its output: {expected}")
        return
    if resume_training:
        run = Path(argv[argv.index("--run-dir") + 1])
        if (run / "run_metadata.json").is_file():
            if (run / "summary.json").is_file():
                # Reconcile an interrupted parent after a successfully completed child.
                state["completed"][name] = {"reconciled": True, "summary": str(run / "summary.json")}
                write(STATE, state)
                return
            if not (run / "last.pt").is_file():
                raise RuntimeError(f"Partial run has no resumable checkpoint: {run}")
            argv = [*argv, "--resume"]
    state.update(status="running", current_stage=name, current_argv=list(map(str, argv)),
                 updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), parent_pid=os.getpid())
    write(STATE, state)
    started = time.perf_counter()
    print(json.dumps({"stage": name, "status": "started"}), flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logfile = LOGS / f"{name}.log"
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    with logfile.open("a", encoding="utf-8") as stream:
        child = subprocess.Popen(list(map(str, argv)), cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        state["child_pid"] = child.pid
        state["current_log"] = str(logfile)
        write(STATE, state)
        code = child.wait()
    if code:
        raise RuntimeError(f"{name} failed with exit {code}; see {logfile}")
    if expected and not Path(expected).is_file():
        raise FileNotFoundError(expected)
    state = read(STATE)
    state["completed"][name] = {"elapsed_seconds": time.perf_counter() - started, "log": str(logfile)}
    write(STATE, state)
    print(json.dumps({"stage": name, "status": "completed", **state["completed"][name]}), flush=True)


def module(name, *args):
    return [PYTHON, "-m", f"docrestore.{name}", *args]


def evaluate(name, manifest, output, mode, group=None, seed=None):
    args = ["--manifest", manifest, "--images-root", DATA, "--output", output,
            "--tessdata-dir", PROJECT / "assets", "--tesseract", TESS,
            "--mode", mode, "--workers", str(read(CONFIG)["ocr"]["workers"]),
            "--oem", "1", "--psm", str(read(CONFIG)["ocr"]["psm"])]
    if group is not None:
        args += ["--method", group, "--seed", str(seed)]
    command(name, module("evaluate", *args), output)


def train(job):
    argv = [PYTHON, *job["argv"][1:], "--device", "cuda", "--tesseract", TESS]
    command(Path(job["run_dir"]).name, argv, Path(job["run_dir"]) / "summary.json", resume_training=True)


def main():
    os.chdir(PROJECT)
    if not STATE.is_file():
        write(STATE, {"status": "starting", "completed": {}, "test_metrics_inspected_before_freeze": False})
    check_lock()
    config = read(CONFIG)
    if not config["configuration_frozen"] or config["calibration_status"] != "VALIDATED_AND_FROZEN":
        raise ValueError("A validated frozen configuration is required")
    # Keep this long experiment awake without altering the system power plan.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        command("main_data", [PYTHON, ROOT / "scripts/generate_main_data.py"], DATA / "eval_manifest.jsonl")
        command("main_validation_diagnostic", module("diagnose", "--data-root", DATA, "--config", CONFIG,
                    "--tessdata-dir", "assets", "--tesseract", TESS, "--output-dir", "results/main_validation_diagnostic"),
                    PROJECT / "results/main_validation_diagnostic/diagnostic.json")
        from docrestore.plan import make_plan
        plan = make_plan(CONFIG, DATA, PROJECT / "assets", RUNS, stage="final")
        write(PROJECT / "provenance/run_plan_phase1.json", plan)
        for job in plan["phase1_candidates"]:
            train(job)
        plan = make_plan(CONFIG, DATA, PROJECT / "assets", RUNS, stage="final", select_from=RUNS)
        write(PROJECT / "provenance/run_plan_selected.json", plan)
        for job in plan["phase2_remaining_seeds"]:
            train(job)
        # Test evaluation begins after all selection and training have finished.
        from docrestore.common import read_jsonl, write_jsonl
        refs = PROJECT / "results/main/references"
        refs.mkdir(parents=True, exist_ok=True)
        write_jsonl(refs / "test_clean_manifest.jsonl", [r for r in read_jsonl(DATA / "manifest.jsonl") if r["split"] == "test"])
        write_jsonl(refs / "test_degraded_manifest.jsonl", [r for r in read_jsonl(DATA / "eval_manifest.jsonl") if r["split"] == "test"])
        clean_output, degraded_output = refs / "clean_ocr.jsonl", refs / "degraded_ocr.jsonl"
        evaluate("test_clean_reference", refs / "test_clean_manifest.jsonl", clean_output, "clean")
        evaluate("test_degraded_reference", refs / "test_degraded_manifest.jsonl", degraded_output, "degraded")
        selected = [job for job in plan["phase1_candidates"] if job["lambda"] == plan["selected_lambda"][job["group"]]]
        selected += plan["phase2_remaining_seeds"]
        results = [clean_output, degraded_output]
        seed0 = {}
        for job in selected:
            tag = Path(job["run_dir"]).name
            output_dir = PROJECT / "results/main/predictions" / tag
            command(f"predict_{tag}", module("train", "predict", "--checkpoint", Path(job["run_dir"]) / "best.pt",
                    "--manifest", DATA / "eval_manifest.jsonl", "--data-root", DATA,
                    "--output-dir", output_dir, "--split", "test", "--device", "cuda"),
                    output_dir / "predictions_manifest.jsonl")
            output = output_dir / "ocr.jsonl"
            evaluate(f"evaluate_{tag}", output_dir / "predictions_manifest.jsonl", output, "restored", job["group"], job["seed"])
            results.append(output)
            if job["seed"] == config["analysis"]["example_seed"]:
                seed0[job["group"]] = output
        aggregate = PROJECT / "results/main/aggregate"
        command("aggregate", module("aggregate", "--results", *results, "--config", CONFIG, "--output-dir", aggregate, "--plot"), aggregate / "summary.json")
        command("cases", module("cases", "--A", seed0["A"], "--B", seed0["B"], "--C", seed0["C"],
                "--degraded", degraded_output, "--config", CONFIG, "--output", PROJECT / "results/main/cases.json"),
                PROJECT / "results/main/cases.json")
        state = read(STATE)
        state.update(status="experiments_complete", current_stage="manuscript", selected_lambda=plan["selected_lambda"],
                     result_summary=str(aggregate / "summary.json"), updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write(STATE, state)
    except BaseException:
        state = read(STATE)
        state.update(status="failed", error=traceback.format_exc(), updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write(STATE, state)
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT))
    main()
