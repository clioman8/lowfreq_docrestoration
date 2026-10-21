"""Preserve supplied original files unchanged inside the study workspace.

Missing originals are recorded, never reconstructed from updated packages.
This script does not read experimental observations or change scientific settings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "analysis/original_inputs"
ORIGINALS = (
    ("original_rough_draft.pdf", "An Empirical Comparison of Low-Frequency Subspace Consistency Regularization for Document Image Restoration_v5_thesis_final.pdf", 228991, None),
    ("original_code_packages.zip", "all_10_github_packages.zip", 256492,
     "d28c5e6042d6b2276b6d67aca331d2f4d097e8c82ad38112886d8936bb541cd1"),
    ("source_database.zip", "An Empirical Comparison of Low-Frequency Subspace Consistency Regularization for Document Image Restoration DB.zip", 1819223408, None),
    ("design_references_only.zip", "선행 연구.zip", 54677212, None),
)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def preserve(source_directory):
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest_path = DESTINATION / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    known = {item["original_filename"]: item for item in previous.get("inputs", [])}
    records = []
    for alias, filename, expected_size, historical_sha in ORIGINALS:
        destination = DESTINATION / filename
        if not destination.resolve().is_relative_to(DESTINATION.resolve()):
            raise ValueError("Original destination is outside its intended directory")
        source = source_directory / filename
        prior_record = known.get(filename, {})
        expected_sha = (prior_record.get("sha256") or prior_record.get("expected_sha256")
                        or historical_sha)
        if destination.is_file():
            current_sha = digest(destination)
            if destination.stat().st_size != expected_size or (expected_sha and current_sha != expected_sha):
                raise ValueError(f"Preserved original changed: {filename}")
        elif source.is_file():
            current_sha = digest(source)
            if source.stat().st_size != expected_size or (expected_sha and current_sha != expected_sha):
                raise ValueError(f"Supplied original differs from its recorded identity: {filename}")
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copyfile(source, temporary)
            if digest(temporary) != current_sha or digest(source) != current_sha:
                raise ValueError(f"Original changed while being preserved: {filename}")
            temporary.replace(destination)
        else:
            records.append({"archive_alias": alias, "original_filename": filename,
                            "status": "missing_exact_original", "expected_bytes": expected_size,
                            "expected_sha256": expected_sha})
            continue
        records.append({"archive_alias": alias, "original_filename": filename, "status": "preserved",
                        "snapshot_path": destination.relative_to(ROOT).as_posix(),
                        "bytes": destination.stat().st_size, "sha256": current_sha})
    result = {"schema_version": 1, "kind": "supplied_original_inputs",
              "checked_at_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
              "inputs": records}
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    result = preserve(args.source_directory)
    missing = [item["original_filename"] for item in result["inputs"] if item["status"] != "preserved"]
    print(json.dumps({"preserved_count": len(result["inputs"]) - len(missing),
                      "missing_originals": missing,
                      "manifest": "analysis/original_inputs/manifest.json"}, ensure_ascii=True))
    if args.require_all and missing:
        raise FileNotFoundError("Exact original input files are still missing: " + ", ".join(missing))


if __name__ == "__main__":
    main()
