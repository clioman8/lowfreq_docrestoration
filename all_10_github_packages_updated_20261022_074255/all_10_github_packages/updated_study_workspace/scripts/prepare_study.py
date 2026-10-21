"""Create development configuration without inspecting final test observations."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "experiment"
config = json.loads((PROJECT / "configs/pilot.example.json").read_text(encoding="utf-8"))
config["data"].update(
    samples_per_split={"train": 512, "validation": 64, "test": 0},
    book_allowlist="assets/english_selection/book_allowlist.json",
    work_id_mapping="assets/english_selection/work_id_mapping.json",
    language_validation=True,
    language_validation_method="Project Gutenberg catalog Language=en; one named primary author; archived metadata and hashes",
)
config["training"].update(max_steps=500, eval_every=250, diagnostics_every=50, cpu_threads=4)
config["ocr"].update(workers=8)
config["notes"] = [
    "Development uses train and validation only; no test images rendered.",
    "Book pool and author groups chosen from source metadata before image generation.",
    "Main data, severity, lambda grid, and training budget will be frozen after validation diagnostics.",
]
destination = PROJECT / "configs/development.json"
destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(destination)
