"""Preserve one dead pre-checkpoint attempt before exact-seed restart.

Use only after confirming both the study parent and training child are absent.
No completed run, scientific source, split, or frozen setting is changed.
"""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "experiment"
RUNS = (PROJECT / "runs/main").resolve()
state_path = PROJECT / "study_status.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
tag = state["current_stage"]
source = (RUNS / tag).resolve()
if source.parent != RUNS or not source.is_relative_to(RUNS):
    raise ValueError("Recovery source is outside the intended runs directory")
if any((source/name).exists() for name in ("summary.json","last.pt","best.pt")):
    raise ValueError("Preserve/resume a real checkpoint instead of restarting this run")
metadata = json.loads((source/"run_metadata.json").read_text(encoding="utf-8"))
lines = (source/"training_log.jsonl").read_text(encoding="utf-8").splitlines()
last_step = json.loads(lines[-1])["step"] if lines else 0
if last_step >= min(metadata["config"]["training"]["eval_steps"]):
    raise ValueError("Attempt progressed beyond the first checkpoint boundary")
now = datetime.now(timezone(timedelta(hours=9)))
destination = (RUNS / "abandoned_precheckpoint" / (tag+"_"+now.strftime("%Y%m%d_%H%M%S"))).resolve()
if not destination.is_relative_to(RUNS) or destination.exists():
    raise ValueError("Unsafe or existing recovery destination")
destination.parent.mkdir(parents=True,exist_ok=True)
# Absolute source/destination have both been checked to remain under RUNS.
source.rename(destination)
event = {"detected_at_kst":now.isoformat(),"stage":tag,"last_logged_step":last_step,
         "reported_parent_pid":state.get("parent_pid"),"reported_child_pid":state.get("child_pid"),
         "process_absence_confirmed":True,"gpu_idle_confirmed":True,
         "preserved_attempt":str(destination),"restart_seed":metadata["seed"],
         "restart_group":metadata["group"],"restart_lambda":metadata["lambda"],
         "scientific_settings_changed":False,"completed_runs_changed":False,
         "reason":"Previous process no longer exists; termination cause is not established. No checkpoint was saved.",
         "timing_caveat":"Main summary times exclude this abandoned pre-checkpoint attempt; retain this recovery record when reporting execution cost."}
event_path = PROJECT/"provenance/recovery_events.json"
events = json.loads(event_path.read_text(encoding="utf-8")) if event_path.exists() else []
events.append(event)
event_path.write_text(json.dumps(events,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
state.update(status="interrupted_recoverable",interruption_event=event,child_pid=None)
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(event,ensure_ascii=True))
