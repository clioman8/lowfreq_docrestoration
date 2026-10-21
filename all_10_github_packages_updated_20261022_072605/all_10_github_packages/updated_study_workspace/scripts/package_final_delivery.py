"""Bundle four original inputs, additional supplied snapshots, and verified final updates.

This packaging step never changes scientific sources or creates missing results.
Run with the bundled PDF Python after full-page visual QA has passed.
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "experiment"
OUTPUT = ROOT / "output/packages"
PDF = ROOT / "output/pdf/low_frequency_subspace_document_restoration.pdf"
QA = ROOT / "output/pdf/final_visual_qa.json"
DOWNLOADS = Path("C:/Users/ParkBS/Downloads")
INPUT_SNAPSHOTS = ROOT / "analysis/input_snapshots"
ORIGINAL_INPUTS = ROOT / "analysis/original_inputs"
INPUTS = [
    ("original_rough_draft.pdf", DOWNLOADS / "An Empirical Comparison of Low-Frequency Subspace Consistency Regularization for Document Image Restoration_v5_thesis_final.pdf"),
    ("original_code_packages.zip", DOWNLOADS / "all_10_github_packages.zip"),
    ("source_database.zip", DOWNLOADS / "An Empirical Comparison of Low-Frequency Subspace Consistency Regularization for Document Image Restoration DB.zip"),
    ("design_references_only.zip", DOWNLOADS / "선행 연구.zip"),
]

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_archive_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Evidence path is outside the study workspace: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.relative_to(ROOT).as_posix()


def supplied_input_snapshots():
    """Resolve immutable reattached inputs before rebuilding any current output ZIP."""
    manifest = INPUT_SNAPSHOTS / "manifest.json"
    if not manifest.exists():
        return []
    records = read(manifest)["snapshots"]
    if not isinstance(records, list):
        raise ValueError("Supplied snapshot manifest must contain a list")
    snapshots, seen = [], set()
    for record in records:
        relative = Path(record["snapshot_path"])
        path = (ROOT / relative).resolve()
        if (relative.is_absolute() or not path.is_relative_to(ROOT.resolve())
                or not path.is_relative_to(INPUT_SNAPSHOTS.resolve())):
            raise ValueError("Supplied snapshot path is outside the intended workspace input directory")
        expected = record["sha256"]
        if (not isinstance(expected, str) or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)):
            raise ValueError("Supplied snapshot has an invalid expected SHA256")
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Supplied input snapshot is missing or changed: {relative}")
        if "bytes" in record and path.stat().st_size != record["bytes"]:
            raise ValueError("Supplied snapshot byte count changed")
        alias = record["archive_alias"]
        if (not isinstance(alias, str) or not alias or alias in {".", ".."}
                or any(char in alias for char in "/\\:")):
            raise ValueError("Supplied snapshot archive alias must be a filename")
        if not record.get("original_filename") or not record.get("received_at_kst"):
            raise ValueError("Supplied snapshot lacks original filename or received timestamp")
        if expected in seen:
            continue
        seen.add(expected)
        archive_path = f"00_original_inputs/additional_supplied_snapshots/{expected}/{alias}"
        snapshots.append((archive_path, path, record))
    return snapshots


def preserved_original_inputs():
    """Use verified workspace copies, retaining the exact-original requirement."""
    manifest_path = ORIGINAL_INPUTS / "manifest.json"
    records = {item["archive_alias"]: item for item in read(manifest_path)["inputs"]} if manifest_path.exists() else {}
    result = []
    for alias, original_source in INPUTS:
        record = records.get(alias)
        if record and record.get("status") == "preserved":
            path = (ROOT / record["snapshot_path"]).resolve()
            if (not path.is_relative_to(ORIGINAL_INPUTS.resolve()) or not path.is_file()
                    or path.name != original_source.name or path.stat().st_size != record["bytes"]
                    or sha(path) != record["sha256"]):
                raise ValueError(f"Preserved original is missing or changed: {original_source.name}")
        else:
            path = original_source
            if not path.is_file():
                raise FileNotFoundError(f"Exact original input is required; do not substitute a reconstructed ZIP: {path}")
            if record and (path.stat().st_size != record["expected_bytes"]
                           or (record.get("expected_sha256") and sha(path) != record["expected_sha256"])):
                raise ValueError(f"Re-supplied original does not match the recorded identity: {path.name}")
        result.append((alias, path))
    return result

def main():
    from pypdf import PdfReader
    state = read(PROJECT / "study_status.json")
    if state["status"] not in {"experiments_complete", "manuscript_complete", "complete"}:
        raise ValueError("Final delivery requires completed experiments")
    quality = read(QA)
    page_count = len(PdfReader(PDF).pages)
    if (quality.get("status") != "passed" or quality.get("pdf_sha256") != sha(PDF)
            or quality.get("page_count") != page_count
            or sorted(quality.get("inspected_pages", [])) != list(range(1, page_count+1))
            or quality.get("issues_remaining")):
        raise ValueError("Final PDF must pass current, all-page visual QA")
    original_inputs = preserved_original_inputs()
    # A reattached code ZIP is an immutable input even when the current output
    # ZIP below is refreshed after manuscript edits.
    snapshot_inputs = supplied_input_snapshots()
    # Rebuild the code snapshot after any manuscript/packaging revisions.
    original_code = dict(original_inputs)["original_code_packages.zip"]
    subprocess.run([sys.executable, str(ROOT / "scripts/package_updated_code.py"),
                    "--original-code", str(original_code)], cwd=ROOT, check=True)
    updated_code = OUTPUT / "all_10_github_packages_updated.zip"
    if not updated_code.is_file():
        raise FileNotFoundError(updated_code)
    aggregate = read(PROJECT / "results/main/aggregate/summary.json")
    plan = read(PROJECT / "provenance/run_plan_selected.json")
    evidence = set()
    for folder in (PROJECT / "provenance", PROJECT / "configs", PROJECT / "assets/english_selection"):
        evidence.update(p for p in folder.rglob("*") if p.is_file())
    evidence.update([PROJECT / "data/main/manifest.jsonl", PROJECT / "data/main/eval_manifest.jsonl",
                     PROJECT / "data/main/split_manifest.json", PROJECT / "data/main/dataset_metadata.json",
                     PROJECT / "data/main/eval_metadata.json", PROJECT / "study_status.json", QA])
    evidence.update(p for p in (PROJECT / "results/main/aggregate").rglob("*") if p.is_file())
    evidence.add(PROJECT / "results/main/cases.json")
    for result in aggregate["aggregation_provenance"]["result_files"]:
        path = Path(result["path"])
        if sha(path) != result["sha256"]:
            raise ValueError("Aggregated OCR evidence changed")
        evidence.add(path)
    for job in plan["phase1_candidates"] + plan["phase2_remaining_seeds"]:
        run = Path(job["run_dir"])
        evidence.update(run / name for name in ("run_metadata.json", "summary.json", "training_log.jsonl", "best_validation_ocr.jsonl"))
    recovery_path = PROJECT / "provenance/recovery_events.json"
    recovery_events = read(recovery_path) if recovery_path.exists() else []
    for event in recovery_events:
        abandoned = Path(event["preserved_attempt"]).resolve()
        if not abandoned.is_relative_to((PROJECT / "runs/main").resolve()):
            raise ValueError("Recovery evidence is outside the intended runs directory")
        evidence.update(abandoned / name for name in ("run_metadata.json", "training_log.jsonl")
                        if (abandoned / name).is_file())
    # Include the actual image panels selected for the manuscript, not all images.
    for case in read(PROJECT / "results/main/cases.json")["cases"].values():
        if case["status"] != "selected":
            continue
        for method in ("input", "A", "B", "C"):
            row = case[method]
            for field in ("clean_path", "degraded_path", "restored_path"):
                if row.get(field):
                    path = Path(row[field])
                    evidence.add(path if path.is_absolute() else PROJECT / "data/main" / path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    evidence_zip = OUTPUT / "reproducibility_evidence.zip"
    evidence_temp = evidence_zip.with_suffix(".zip.tmp")
    evidence_manifest = []
    with zipfile.ZipFile(evidence_temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(evidence, key=lambda p: str(p)):
            relative = safe_archive_path(path)
            archive.write(path, relative)
            evidence_manifest.append({"path":relative, "bytes":path.stat().st_size, "sha256":sha(path)})
        archive.writestr("EVIDENCE_MANIFEST.json", json.dumps(evidence_manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(evidence_temp) as archive:
        if archive.testzip() is not None:
            raise ValueError("Reproducibility archive CRC verification failed")
    evidence_temp.replace(evidence_zip)
    entries = [(f"00_original_inputs/{alias}", path, "supplied_original_unchanged") for alias,path in original_inputs]
    entries += [(name, path, "additional_supplied_code_snapshot_unchanged")
                for name, path, _ in snapshot_inputs]
    entries += [("01_updated/completed_thesis.pdf", PDF, "completed_and_visually_verified"),
                ("01_updated/all_10_github_packages_updated.zip", updated_code, "updated_code_and_manuscript_sources"),
                ("02_reproducibility/reproducibility_evidence.zip", evidence_zip, "measured_results_and_provenance")]
    manifest = {"created_at_kst":datetime.now(timezone(timedelta(hours=9))).isoformat(),
                "original_inputs_preserved":True, "final_pdf_pages":page_count,
                "original_supplied_input_count":len(INPUTS),
                "additional_supplied_snapshot_count":len(snapshot_inputs),
                "additional_supplied_snapshots":[{**record, "archive_path":name}
                                                  for name, _, record in snapshot_inputs],
                "design_references_used_for_design_only":True,
                "entries":[{"archive_path":name,"source_name":path.name,"bytes":path.stat().st_size,
                            "sha256":sha(path),"status":kind} for name,path,kind in entries],
                "recovery_events_included":len(recovery_events),
                "execution_timing_note":"Completed-run summary times exclude any abandoned pre-checkpoint attempts, which are preserved separately in the evidence archive.",
                "large_reconstruction_note":"Full generated images, installed libraries, and model checkpoints are not duplicated; their regeneration uses the included frozen protocol and original source database."}
    delivery = OUTPUT / "complete_thesis_delivery.zip"
    delivery_temp = delivery.with_suffix(".zip.tmp")
    with zipfile.ZipFile(delivery_temp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name,path,_ in entries:
            archive.write(path,name)
        archive.writestr("DELIVERY_MANIFEST.json",json.dumps(manifest,ensure_ascii=False,indent=2))
        archive.writestr("READ_ME.txt",
            "Completed document restoration thesis delivery\n\n"
            f"00_original_inputs contains the four original supplied inputs plus {len(snapshot_inputs)} additional supplied code snapshot(s), byte-for-byte unchanged.\n"
            "Additional supplied snapshots are preserved under additional_supplied_snapshots/ by their immutable SHA256; they are distinct from the final refreshed code ZIP in 01_updated.\n"
            "Short archive names map to the original names and SHA256 checksums in DELIVERY_MANIFEST.json.\n"
            "The manifest also records each additional snapshot's original filename, received timestamp, relative source path, and expected SHA256.\n"
            "01_updated contains the completed thesis and the updated ten code packages, including current study scripts.\n"
            "02_reproducibility contains actual OCR records, numerical/test/data validation, frozen settings, metric tables, and selected case images.\n"
            "The reference-paper archive was used for layout and design only.\n"
            "Generated full-image datasets, installed Python/CUDA libraries, and model checkpoints are not duplicated in this delivery.\n"
            "No source input in Downloads was overwritten.\n")
    with zipfile.ZipFile(delivery_temp) as archive:
        if archive.testzip() is not None:
            raise ValueError("Final delivery CRC verification failed")
        for name, _, record in snapshot_inputs:
            if hashlib.sha256(archive.read(name)).hexdigest() != record["sha256"]:
                raise ValueError("Archived additional supplied snapshot failed expected SHA256 verification")
    delivery_temp.replace(delivery)
    manifest["delivery_archive"] = {"path":str(delivery),"bytes":delivery.stat().st_size,"sha256":sha(delivery)}
    (OUTPUT / "delivery_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"delivery":str(delivery),"bytes":delivery.stat().st_size,"evidence":str(evidence_zip),
                      "updated_code":str(updated_code),"final_pdf":str(PDF)},ensure_ascii=True))

if __name__ == "__main__":
    main()
