"""Freeze measured development choices before final test image generation."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "experiment"
def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

destination = PROJECT / "configs/frozen.json"
if destination.exists() or (PROJECT / "data/main/manifest.jsonl").exists():
    raise FileExistsError("Never overwrite a frozen study or refreeze after final data generation")
pilots = {}
for tag in ("A_lambda0_seed0", "B_lambda1_seed0", "C_lambda1_seed0", "B_lambda10_seed0", "C_lambda10_seed0"):
    pilots[tag] = read(PROJECT / "runs/development" / tag / "summary.json")
diagnostic = read(PROJECT / "results/development_diagnostic/diagnostic.json")
assert diagnostic["test_inspected"] is False
assert read(PROJECT / "data/development/dataset_metadata.json")["counts"]["test"] == 0
config = read(PROJECT / "configs/development.json")
config["configuration_frozen"] = True
config["calibration_status"] = "VALIDATED_AND_FROZEN"
config["data"].update(samples_per_split={"train":8000,"validation":1000,"test":1000},
                      source_split_manifest="data/development/split_manifest.json")
config["loss"]["lambda_grid"] = [.1, 1., 10.]
config["training"].update(max_steps=10000, eval_every=1000,
    eval_steps=[250,500,1000,2000,4000,6000,8000,10000], cpu_threads=4)
config["ocr"]["workers"] = 16
config["notes"] = [
    "Frozen after train/validation development; no test images or test OCR inspected.",
    "Severity path retained after clean/input validation OCR and image inspection.",
    "Identical three-value lambda budgets for B/C; gradient measurements motivate 0.1, 1, 10.",
    "Shared checkpoint schedule includes early updates following pilot validation behavior.",
    "Three matched training seeds; selected seed-0 candidate reused as the first repetition.",
    "Source author groups and data split are reused unchanged from development.",
]
write(destination, config)
lock = {
    "frozen_at_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
    "config_sha256": sha(destination),
    "source_sha256": {p.relative_to(PROJECT).as_posix():sha(p) for p in sorted((PROJECT/"docrestore").glob("*.py"))},
    "test_images_rendered_before_freeze": False,
    "test_ocr_inspected_before_freeze": False,
    "development_diagnostic": diagnostic,
    "development_pilots": pilots,
    "source_split_sha256": sha(PROJECT / "data/development/split_manifest.json"),
    "main_training_runs":13,
    "reported_training_repetitions":9,
    "checkpoint_evaluations_per_run":8,
    "planned_training_updates":130000,
    "planned_validation_ocr_images":520000,
    "selection_policy":"max shifted corpus CER, then mean shifted CER, then smaller lambda; earliest checkpoint on exact ties",
    "platform":platform.platform(),
    "python":sys.version,
    "selection_assets":{p.name:sha(p) for p in (PROJECT/"assets/english_selection").glob("*.json")},
}
write(PROJECT/"provenance/study_lock.json", lock)
requirements = subprocess.check_output([sys.executable,"-m","pip","freeze"], text=True)
(PROJECT/"provenance/environment_requirements.txt").write_text(requirements,encoding="utf-8")
print(json.dumps({"frozen_config":str(destination),"sha256":lock["config_sha256"],"frozen_at_kst":lock["frozen_at_kst"]}))
