"""Verify generated files, correspondence, and split isolation without OCR."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "experiment"
DATA = PROJECT / "data/main"

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def verify():
    config = read(PROJECT / "configs/frozen.json")
    clean = rows(DATA / "manifest.jsonl")
    evaluation = rows(DATA / "eval_manifest.jsonl")
    split = read(DATA / "split_manifest.json")
    expected = config["data"]["samples_per_split"]
    if Counter(r["split"] for r in clean) != Counter(expected):
        raise ValueError("Clean sample counts differ from the frozen design")
    severities = {s["id"]: s for s in config["degradation"]["severities"]}
    expected_eval = {(part, severity):expected[part] for part in ("validation", "test") for severity in severities}
    if Counter((r["split"], r["severity_id"]) for r in evaluation) != Counter(expected_eval):
        raise ValueError("Incomplete evaluation manifest")
    index = {r["sample_id"]:r for r in clean}
    if len(index) != len(clean):
        raise ValueError("Duplicate clean sample ID")
    group_splits, transcript_splits, image_splits = {}, {}, {}
    used_books = {part:set() for part in expected}
    checked_files = 0
    for r in clean:
        if split["book_assignments"][r["book_id"]] != r["split"]:
            raise ValueError("Book crosses its frozen source partition")
        used_books[r["split"]].add(r["book_id"])
        for mapping, key in ((group_splits,r["source_group_id"]),
                             (transcript_splits,r["gt_normalized_sha256"]),
                             (image_splits,r["clean_sha256"])):
            if mapping.setdefault(key,r["split"]) != r["split"]:
                raise ValueError("Source, transcript, or image hash crosses splits")
        if sha(DATA/r["clean_path"]) != r["clean_sha256"]:
            raise ValueError("Clean image hash mismatch")
        checked_files += 1
    keys = set()
    for r in evaluation:
        key = (r["sample_id"],r["severity_id"])
        if key in keys:
            raise ValueError("Duplicate sample/severity key")
        keys.add(key)
        original = index[r["sample_id"]]
        for field in ("split","book_id","source_group_id","gt_text","gt_normalized_sha256","clean_sha256"):
            if original[field] != r[field]:
                raise ValueError("Evaluation target correspondence mismatch")
        spec = severities[r["severity_id"]]
        expected_eta = {k:float(spec[k]) for k in ("kappa","alpha","T")}
        expected_eta.update(n_steps=config["degradation"]["n_steps"],h=config["degradation"]["h"])
        if r["eta"] != expected_eta:
            raise ValueError("Stored severity differs from frozen parameters")
        for file_key, hash_key in (("degraded_path","degraded_sha256"),("degraded_array_path","degraded_array_sha256")):
            if sha(DATA/r[file_key]) != r[hash_key]:
                raise ValueError("Degraded image/array hash mismatch")
            checked_files += 1
    result = {"status":"passed", "test_ocr_inspected":False,
              "clean_counts":dict(Counter(r["split"] for r in clean)),
              "evaluation_counts":{f"{p}_{s}":n for (p,s),n in expected_eval.items()},
              "used_source_book_counts":{p:len(ids) for p,ids in used_books.items()},
              "file_hashes_verified":checked_files,
              "cross_split_source_groups":0,"cross_split_exact_transcripts":0,"cross_split_exact_images":0,
              "manifest_sha256":sha(DATA/"manifest.jsonl"),
              "eval_manifest_sha256":sha(DATA/"eval_manifest.jsonl")}
    target = PROJECT/"provenance/data_integrity.json"
    target.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result

if __name__ == "__main__":
    print(json.dumps(verify(),indent=2))
