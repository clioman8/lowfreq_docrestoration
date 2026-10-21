"""Package current code without changing frozen scientific sources or reading OCR results.

The ten original nested ZIP names are retained. A small integrated workspace
contains code and provenance, while generated observations/results and large
input assets remain outside this code-only deliverable. Re-run before final
delivery to capture later manuscript/orchestration changes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "experiment"
DEFAULT_ORIGINAL = Path.home() / "Downloads/all_10_github_packages.zip"
ORIGINAL_CODE_BYTES = 256492
ORIGINAL_CODE_SHA256 = "d28c5e6042d6b2276b6d67aca331d2f4d097e8c82ad38112886d8936bb541cd1"
PREFIX = "all_10_github_packages/"
PROVENANCE = ("study_lock.json", "test_validation.json", "test_validation.log",
              "numerical_verification.json", "data_integrity.json", "environment_requirements.txt")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def zip_bytes(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in sorted(entries.items()):
            require(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts,
                    f"Unsafe archive path: {name}")
            archive.writestr(name, value)
    return output.getvalue()


def source_integrity():
    lock = json.loads((PROJECT / "provenance/study_lock.json").read_text(encoding="utf-8"))
    tested = json.loads((PROJECT / "provenance/test_validation.json").read_text(encoding="utf-8"))
    require(sha((PROJECT / "configs/frozen.json").read_bytes()) == lock["config_sha256"],
            "Frozen configuration differs from the study lock")
    for relative, expected in lock["source_sha256"].items():
        require(sha((PROJECT / relative).read_bytes()) == expected, f"Frozen source changed: {relative}")
        require(tested["source_sha256"].get(relative) == expected, f"Source differs from tested version: {relative}")
    return lock, tested


def original_members(original_code=None, original_code_directory=None):
    """Load a raw original or explicitly verify recovered extracted members."""
    recovered = original_code_directory is not None
    historical_sha = None
    evidence = None
    if recovered:
        directory = Path(original_code_directory).resolve()
        old_index_bytes = (directory / "PACKAGE_INDEX.json").read_bytes()
        old_readme = (directory / "READ_FIRST.md").read_bytes()
        manifest_path = ROOT / "analysis/input_snapshots/manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        require(bool(manifest.get("snapshots")), "No immutable first-snapshot provenance is available")
        first = manifest["snapshots"][0]
        snapshot_path = ROOT / first["snapshot_path"]
        snapshot_bytes = snapshot_path.read_bytes()
        require(len(snapshot_bytes) == first["bytes"] and sha(snapshot_bytes) == first["sha256"],
                "Immutable first snapshot differs from its input manifest")
        with zipfile.ZipFile(io.BytesIO(snapshot_bytes)) as archive:
            require(archive.testzip() is None, "Immutable first snapshot CRC failed")
            require(old_index_bytes == archive.read(PREFIX + "original_metadata/PACKAGE_INDEX.original.json"),
                    "Extracted original index differs from first-snapshot original_metadata")
            require(old_readme == archive.read(PREFIX + "original_metadata/READ_FIRST.original.md"),
                    "Extracted original READ_FIRST differs from first-snapshot original_metadata")
            historical_sha = json.loads(archive.read(PREFIX + "CHANGELOG.json")).get(
                "snapshot", {}).get("original_code_zip_sha256")
        evidence = {"input_snapshot_manifest": str(manifest_path.resolve()),
                    "input_snapshot_manifest_sha256": sha(manifest_bytes),
                    "first_snapshot": str(snapshot_path.resolve()),
                    "first_snapshot_sha256": first["sha256"],
                    "original_index_sha256": sha(old_index_bytes),
                    "original_readme_sha256": sha(old_readme)}
        original_sha = None
        basis = "verified_extracted_original_members"
    else:
        archive_path = Path(original_code or DEFAULT_ORIGINAL)
        original_bytes = archive_path.read_bytes()
        original_sha = sha(original_bytes)
        require(len(original_bytes) == ORIGINAL_CODE_BYTES and original_sha == ORIGINAL_CODE_SHA256,
                "Raw original code ZIP differs from the known exact supplied archive; "
                "an updated or reconstructed ZIP must not be labeled unchanged original")
        basis = "raw_original_code_archive"
        with zipfile.ZipFile(io.BytesIO(original_bytes)) as archive:
            require(archive.testzip() is None, "Original outer ZIP CRC failed")
            old_index_bytes = archive.read(PREFIX + "PACKAGE_INDEX.json")
            old_readme = archive.read(PREFIX + "READ_FIRST.md")
    old_index = json.loads(old_index_bytes)
    require(isinstance(old_index, list) and len(old_index) == 10,
            "The original package index must contain exactly ten nested ZIPs")
    members = {}
    for item in old_index:
        name = item["file"]
        require(isinstance(name, str) and PurePosixPath(name).name == name and
                "\\" not in name and name.endswith(".zip") and name not in members,
                f"Invalid or duplicate original nested ZIP name: {name}")
        if recovered:
            blob = (directory / name).read_bytes()
        else:
            with zipfile.ZipFile(io.BytesIO(original_bytes)) as archive:
                blob = archive.read(PREFIX + name)
        require(len(blob) == item["bytes"] and sha(blob) == item["sha256"],
                f"Original nested ZIP differs from PACKAGE_INDEX: {name}")
        with zipfile.ZipFile(io.BytesIO(blob)) as nested:
            require(nested.testzip() is None, f"Original nested CRC failed: {name}")
        members[name] = blob
    provenance = {"packaging_basis": basis, "original_code_zip_sha256": original_sha,
                  "historically_recorded_original_code_zip_sha256": historical_sha,
                  "original_code_archive_unchanged": None if recovered else True,
                  "original_members_verified": True,
                  "original_outer_archive_presently_unavailable": recovered,
                  "original_member_verification": evidence}
    return old_index_bytes, old_readme, members, provenance


def original_archive_notice(snapshot):
    if snapshot["packaging_basis"] != "verified_extracted_original_members":
        return "Packaging basis: the raw original code archive, verified without changing it."
    return ("Packaging basis: verified extracted original members. The original outer "
            "all_10_github_packages.zip is presently unavailable. The ten nested ZIPs were "
            "verified against PACKAGE_INDEX, and the original index/READ_FIRST bytes were "
            "verified against original_metadata in the SHA-verified immutable first snapshot. "
            "No original outer archive was reconstructed or verified as unchanged. The historical "
            "outer SHA, if recorded, is provenance only. A final bundle claiming all four unchanged "
            "original attachments still requires the exact missing original outer ZIP.")


def current_notice(snapshot):
    return f"""# Updated code snapshot

Packaged {snapshot['packaged_at_kst']}. Snapshot kind: **code and reproducibility materials**.
Study state at packaging: **{snapshot['study_status']}** / {snapshot['current_stage']}.
{original_archive_notice(snapshot)}

This ZIP contains no main training checkpoints, OCR result files, generated images,
final performance tables, or final thesis PDF. The original DB ZIP and design-reference
ZIP remain unchanged and are supplied separately. Research contents from the design
reference ZIP are not incorporated into this package.

The original ten nested ZIP names and common lowfreq_docrestoration/ layout are retained.
Current duplicate modules have identical bytes. Nested ZIPs remain stage-specific;
merge all ten for the full pipeline or use updated_study_workspace/ directly.

validation/*.json retains the original historical packaging/smoke records. These are
not current performance findings. validation/latest/ contains the separate merged-project
software record: {snapshot['tests_run']} tests passed, including real CUDA backward and
exact CPU checkpoint-resume verification. It does not claim that every isolated nested
ZIP was independently re-tested, or that the main comparison has completed.

Current changes include metadata-described English source verification, ordered parallel
OCR (--workers), strict deterministic training, official Tesseract build version matching,
exact opt-in --resume with Adam/RNG/loader recovery, atomic checkpoints, explicit
training.eval_steps, Windows UTF8/font test portability, evidence-gated manuscript
generation, seed-wise secondary mean shifted CER reporting, and qualified contributor/work
group terminology. Original starter guides describe earlier example defaults; the frozen JSON
and this update guide take precedence for the archived study.

Frozen main configuration: 8,000/1,000/1,000 train/validation/test regions, 10,000 updates,
seeds 0/1/2, equal B/C lambda candidates [0.1,1,10], checkpoints at
[250,500,1000,2000,4000,6000,8000,10000], and 16 OCR workers. The protocol plans 13 completed
training runs and 9 selected reported repetitions; selected seed-0 tuning runs are reused.
These are protocol settings, not observed performance results.

See updated_study_workspace/REPRODUCTION_README.md, CHANGELOG.json,
CURRENT_SOURCE_SHA256.json and ARCHIVE_CONTENTS_SHA256.json for reproducibility and hashes.
"""


def reproduction_readme(snapshot):
    return f"""# Reproduction workspace — code snapshot

Snapshot: {snapshot['packaged_at_kst']}; study state: {snapshot['study_status']} /
{snapshot['current_stage']}. No final experimental findings or completed PDF are in this code ZIP.
{original_archive_notice(snapshot)}

The completion handoff is an archival record: its process IDs, session IDs, automation status,
absolute paths and heartbeat timestamps are not live instructions on another machine.

This folder mirrors the original workspace: scripts/, analysis/, experiment/. Main code,
all five test modules, the actual frozen configuration, source selection metadata, source
partition metadata and current implementation/data-integrity checks are included. Heavy input
   assets, Tesseract installation, .venv, all generated image/observation data, active process state, runs, results,
predictions, images, caches and final PDF are deliberately omitted. source_partition/ contains
only the pre-existing book-group split metadata, not rendered observations or OCR results.
The single experiment/data/development/split_manifest.json file is also included at the exact
relative path referenced by frozen.json; it contains source-group assignments, not image data.

On Windows, from this folder:

1. Create .venv with Python 3.13 and install experiment/requirements.txt. The recorded tested
   environment is experiment/provenance/environment_requirements.txt; the tested GPU build is
   PyTorch 2.10.0+cu128 on RTX 5070 Ti. Select a compatible CUDA build for the target hardware.
   PDF creation additionally needs reportlab and pypdf; matplotlib and Pillow are already listed.
2. Extract the user's unchanged DB ZIP outside this folder. Prepare its corpus TGZ, Liberation
   2.1.5 font TAR.GZ and eng.traineddata using the included input-preparation helper:

   .venv\\Scripts\\python.exe scripts\\prepare_supplied_inputs.py --corpus CORPUS_TGZ --font-archive FONT_TAR_GZ --eng ENG_TRAINEDDATA

   It verifies official corpus/model identifiers, restores exactly the 500 selected source texts,
   preserves the three Regular font files and license, and restores the frozen source split.
   It does not render images, train, inspect OCR or change scientific code/configuration.
3. Place the full Tesseract 5.5.3 Windows installation (including its DLLs) under
   experiment/tools/tesseract/, so tools/tesseract/tesseract.exe exists relative to experiment/.
   The tested official build was v5.5.3.20260724. Keep the archived eng.traineddata fixed.
4. Preserve configs/frozen.json and provenance/study_lock.json. Do not run freeze_study.py
   over the archived frozen configuration. prepare_study.py and freeze_study.py are retained
   for provenance and for a separately designed study, not for changing this frozen one.
5. For tests, run from experiment/ with the virtual environment Python. On the restricted
   Windows sandbox used here, set tempfile.tempdir to a workspace-owned scratch folder first.
   The archived test_validation record describes the full merged suite, not isolated packages.
6. To reproduce the frozen main pipeline from this folder, after supplying omitted inputs:

   .venv\\Scripts\\python.exe -u scripts\\run_study.py

   It checks frozen hashes, generates main observations, performs validation-only diagnostics,
   trains seven seed-0 candidates then six selected-seed runs, and only then evaluates held-out
   test documents and generates aggregate/case records. Do not launch a duplicate pipeline
   against an already running original workspace. The code snapshot contains no active status
   file or checkpoints and cannot resume the original machine's in-progress execution.
7. After all results exist, build_manuscript.py --check-inputs verifies evidence. PDF authoring
   also needs the complete results and Times fonts, then every PDF page must be rendered and
   visually reviewed. The included analysis/manuscript_sections.json contains methodological
   prose; empirical abstract/results/discussion/conclusion are generated from completed records.

Frozen absolute paths recorded inside historical provenance identify the original machine.
The runnable scripts derive their workspace from __file__; the restored source-partition file
is copied to the relative path referenced by frozen.json. Archived generated-data hashes refer
to the original study and are evidence, not files supplied in this code-only package.

The ten stage-specific ZIPs can instead be merged into lowfreq_docrestoration/. All duplicated
modules match the current frozen version. Original historical validation objects are retained
byte-for-byte; current records live under validation/latest/ or experiment/provenance/.
The final delivery bundle is assembled separately by scripts/package_final_delivery.py when
available; it may add the unchanged original four attachments, reviewed PDF and final evidence.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    original = parser.add_mutually_exclusive_group()
    original.add_argument("--original-code", type=Path,
                          help="Raw original outer code ZIP (default: Downloads/all_10_github_packages.zip)")
    original.add_argument("--original-code-directory", type=Path,
                          help="Explicit recovery mode: verify extracted original members against the immutable first snapshot")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/packages/all_10_github_packages_updated.zip")
    args = parser.parse_args()
    old_index_bytes, old_readme, original_nested, original_provenance = original_members(
        args.original_code, args.original_code_directory)
    lock, tested = source_integrity()
    status = json.loads((PROJECT / "study_status.json").read_text(encoding="utf-8"))
    snapshot = {"packaged_at_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
                "kind": "current_code_snapshot_no_main_results", "study_status": status.get("status"),
                "current_stage": status.get("current_stage"), "tests_run": tested["tests_run"],
                "main_results_included": False, "final_pdf_included": False,
                **original_provenance, "frozen_config_sha256": lock["config_sha256"]}
    notice = current_notice(snapshot).encode("utf-8")
    old_index = json.loads(old_index_bytes)
    current = {}
    for folder in ("docrestore", "tests", "configs", "docs", "requirements"):
        for path in sorted((PROJECT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                current[path.relative_to(PROJECT).as_posix()] = path.read_bytes()
    for path in sorted(PROJECT.glob("*")):
        if path.is_file() and (path.suffix in {".md", ".toml", ".txt", ".ps1"}):
            current[path.name] = path.read_bytes()
    latest = {name: (PROJECT / "provenance" / name).read_bytes() for name in PROVENANCE}
    changes, package_entries, index = [], {}, []
    duplicate_modules = {}
    for item in old_index:
        name = item["file"]
        old_nested = original_nested[name]
        source_zip = zipfile.ZipFile(io.BytesIO(old_nested))
        require(source_zip.testzip() is None, f"Original nested CRC failed: {name}")
        entries = {info.filename: source_zip.read(info.filename) for info in source_zip.infolist()
                   if not info.is_dir()}
        package_id = Path(name).stem
        local_changes = []
        for archive_name, previous in list(entries.items()):
            relative = archive_name.removeprefix("lowfreq_docrestoration/")
            if relative in current and current[relative] != previous:
                entries[archive_name] = current[relative]
                local_changes.append({"path": relative, "change": "updated_from_current_workspace",
                                      "old_sha256": sha(previous), "sha256": sha(current[relative])})
        # Include the parallel evaluator's test dependencies wherever evaluate.py is shipped.
        if "lowfreq_docrestoration/docrestore/evaluate.py" in entries:
            for relative in ("tests/test_evaluate_parallel.py", "docrestore/diagnose.py", "docrestore/common.py"):
                destination = "lowfreq_docrestoration/" + relative
                if destination not in entries:
                    entries[destination] = current[relative]
                    local_changes.append({"path": relative, "change": "added_parallel_ocr_test_or_dependency",
                                          "sha256": sha(current[relative])})
        for relative, value in current.items():
            if relative.startswith("configs/"):
                entries["lowfreq_docrestoration/" + relative] = value
        entries["lowfreq_docrestoration/docs/UPDATED_REPRODUCTION.md"] = notice
        entries["lowfreq_docrestoration/README.md"] = notice + b"\n---\n\n" + current["README.md"]
        entries["lowfreq_docrestoration/validation/HISTORICAL_NOTICE.md"] = (
            b"Original validation/*.json objects are retained byte-for-byte and describe historical "
            b"packaging/fixture checks, not updated thesis results. See latest/ for current merged-project checks.\n")
        for filename in ("test_validation.json", "numerical_verification.json"):
            entries["lowfreq_docrestoration/validation/latest/" + filename] = latest[filename]
        entries["lowfreq_docrestoration/validation/latest/README.md"] = (
            b"These records describe the full merged project, not separate re-testing of this isolated ZIP. "
            b"They verify software/numerics and do not contain final restoration-performance findings.\n")
        pack_path = f"lowfreq_docrestoration/packaging/{package_id}.json"
        if pack_path in entries:
            previous_pack = entries[pack_path]
            entries[f"lowfreq_docrestoration/packaging/historical/{package_id}.original.json"] = previous_pack
            package_metadata = json.loads(previous_pack)
            package_metadata["source_files_sha256"] = {
                path.removeprefix("lowfreq_docrestoration/"): sha(value)
                for path, value in entries.items() if "/docrestore/" in path and path.endswith(".py")}
            package_metadata.update(updated_snapshot=snapshot,
                                    latest_validation_scope="merged project software verification; not scientific results",
                                    scientific_results_included=False)
            entries[pack_path] = encoded(package_metadata)
        for path, value in entries.items():
            if "/docrestore/" in path and path.endswith(".py"):
                relative = path.split("/docrestore/", 1)[1]
                require(relative not in duplicate_modules or duplicate_modules[relative] == sha(value),
                        f"Duplicate module version mismatch: {relative}")
                duplicate_modules[relative] = sha(value)
        blob = zip_bytes(entries)
        package_entries[PREFIX + name] = blob
        index.append({**item, "bytes": len(blob), "sha256": sha(blob),
                      "original_bytes": len(old_nested), "original_sha256": sha(old_nested),
                      "entries": len(entries), "snapshot_kind": snapshot["kind"]})
        changes.append({"package": name, "updated_existing_files": local_changes,
                        "current_config_files": sorted(path for path in current if path.startswith("configs/")),
                        "historical_validation_preserved": True})
    package_entries[PREFIX + "original_metadata/PACKAGE_INDEX.original.json"] = old_index_bytes
    package_entries[PREFIX + "original_metadata/READ_FIRST.original.md"] = old_readme
    package_entries[PREFIX + "PACKAGE_INDEX.json"] = encoded(index)
    package_entries[PREFIX + "READ_FIRST.md"] = notice + b"\n---\n\n" + old_readme
    package_entries[PREFIX + "CHANGELOG.json"] = encoded({"snapshot": snapshot, "packages": changes})
    package_entries[PREFIX + "CURRENT_SOURCE_SHA256.json"] = encoded(lock["source_sha256"])
    workspace_prefix = PREFIX + "updated_study_workspace/"
    for relative, value in current.items():
        package_entries[workspace_prefix + "experiment/" + relative] = value
    for path in sorted((ROOT / "scripts").iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".py", ".ps1"}:
            continue
        package_entries[workspace_prefix + "scripts/" + path.name] = path.read_bytes()
    for name in ("manuscript_sections.json", "completion_handoff.json", "continuation_prompt_ko.txt",
                 "code_audit.md", "db_audit.md"):
        path = ROOT / "analysis" / name
        if path.is_file():
            package_entries[workspace_prefix + "analysis/" + name] = path.read_bytes()
    for name, value in latest.items():
        package_entries[workspace_prefix + "experiment/provenance/" + name] = value
    split = PROJECT / "data/development/split_manifest.json"
    require(sha(split.read_bytes()) == lock["source_split_sha256"], "Source partition changed")
    package_entries[workspace_prefix + "experiment/provenance/source_partition/development_split_manifest.json"] = split.read_bytes()
    package_entries[workspace_prefix + "experiment/data/development/split_manifest.json"] = split.read_bytes()
    for path in sorted((PROJECT / "assets/english_selection").glob("*.json")):
        require(sha(path.read_bytes()) == lock["selection_assets"][path.name], f"Source selection changed: {path.name}")
        package_entries[workspace_prefix + "experiment/assets/english_selection/" + path.name] = path.read_bytes()
    included_data = [name for name in package_entries if "/experiment/data/" in name]
    require(included_data == [workspace_prefix + "experiment/data/development/split_manifest.json"],
            "Only the frozen development split metadata may be included from experiment/data")
    package_entries[workspace_prefix + "REPRODUCTION_README.md"] = reproduction_readme(snapshot).encode("utf-8")
    package_entries[workspace_prefix + "experiment/docs/UPDATED_REPRODUCTION.md"] = notice
    package_entries[PREFIX + "ARCHIVE_CONTENTS_SHA256.json"] = encoded({
        "scope": "All outer entries except this self-referential manifest; nested ZIP hashes bind their contents",
        "sha256": {name: sha(value) for name, value in sorted(package_entries.items())}})
    result = zip_bytes(package_entries)
    verified_nested = 0
    with zipfile.ZipFile(io.BytesIO(result)) as archive:
        require(archive.testzip() is None, "Updated outer ZIP CRC failed")
        for item in index:
            blob = archive.read(PREFIX + item["file"])
            require(sha(blob) == item["sha256"], "Nested ZIP hash mismatch")
            with zipfile.ZipFile(io.BytesIO(blob)) as nested:
                require(nested.testzip() is None, "Updated nested ZIP CRC failed")
            verified_nested += 1
        hashes = json.loads(archive.read(PREFIX + "ARCHIVE_CONTENTS_SHA256.json"))["sha256"]
        require(all(sha(archive.read(name)) == value for name, value in hashes.items()), "Outer content hash failed")
    source_integrity()
    rechecked_index, rechecked_readme, rechecked_members, rechecked_provenance = original_members(
        args.original_code, args.original_code_directory)
    require((rechecked_index, rechecked_readme, rechecked_provenance) ==
            (old_index_bytes, old_readme, original_provenance) and
            {name: sha(value) for name, value in rechecked_members.items()} ==
            {name: sha(value) for name, value in original_nested.items()},
            "Original packaging inputs changed during packaging")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(result)
    temporary.replace(args.output)
    report = {"output": str(args.output.resolve()), "bytes": len(result), "sha256": sha(result),
              "snapshot": snapshot, "outer_entries": len(package_entries),
              "nested_zip_crc_passed": verified_nested, "outer_crc_passed": True,
              "outer_content_sha256_verified": len(hashes),
              "duplicate_module_versions_consistent": len(duplicate_modules),
              "allowed_data_metadata_files": ["experiment/data/development/split_manifest.json"],
              **original_provenance, "frozen_source_and_tested_hashes_match": True,
              "omitted": ["main runs/results/checkpoints/OCR", "generated observations and images",
                          "venv/cache", "large DB/font/OCR assets", "original design-reference contents", "final PDF"]}
    args.output.with_suffix(".validation.json").write_bytes(encoded(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
