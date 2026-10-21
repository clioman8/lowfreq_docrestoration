"""Recover deterministic final data generation after an interrupted evaluation pass."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "experiment"
sys.path.insert(0, str(PROJECT))
from docrestore.common import config_hash, read_jsonl
from docrestore.data import build_dataset, prepare_eval_dataset

config = json.loads((PROJECT / "configs/frozen.json").read_text(encoding="utf-8"))
root = PROJECT / "data/main"
manifest = root / "manifest.jsonl"
if manifest.exists():
    metadata = json.loads((root / "dataset_metadata.json").read_text(encoding="utf-8"))
    if metadata["config_sha256"] != config_hash(config):
        raise ValueError("Existing main data uses a different frozen configuration")
    rows = read_jsonl(manifest)
else:
    rows = build_dataset(PROJECT / "assets/english_books", PROJECT / "assets/fonts", root, config)
if not (root / "eval_manifest.jsonl").exists():
    prepare_eval_dataset(rows, root, config)
print(json.dumps({"clean_samples": len(rows), "evaluation_samples": len(read_jsonl(root / "eval_manifest.jsonl"))}))
