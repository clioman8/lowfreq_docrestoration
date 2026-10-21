"""Validation-only clean/degraded OCR sanity check before restoration training."""
from __future__ import annotations
import argparse
from pathlib import Path
import time

from .common import read_json, read_jsonl, write_json, write_jsonl
from .evaluate import evaluate_manifest
from .metrics import corpus_cer


def run_diagnostic(data_root, config, tessdata_dir, output_dir, tesseract="tesseract", *,
                   workers=None):
    root, out = Path(data_root), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean = [r for r in read_jsonl(root / "manifest.jsonl") if r["split"] == "validation"]
    degraded = [r for r in read_jsonl(root / "eval_manifest.jsonl") if r["split"] == "validation"]
    if not clean or not degraded:
        raise ValueError("Generate separate validation images first")
    write_jsonl(out / "clean_manifest.jsonl", clean)
    write_jsonl(out / "degraded_manifest.jsonl", degraded)
    psm = config.get("ocr", {}).get("psm", 6)
    workers = config.get("ocr", {}).get("workers", 1) if workers is None else workers
    start = time.perf_counter()
    evaluate_manifest(out / "clean_manifest.jsonl", root, out / "clean_ocr.jsonl", tessdata_dir,
                      tesseract=tesseract, mode="clean", method="clean", psm=psm, oem=1,
                      workers=workers)
    evaluate_manifest(out / "degraded_manifest.jsonl", root, out / "degraded_ocr.jsonl", tessdata_dir,
                      tesseract=tesseract, mode="degraded", method="degraded", psm=psm, oem=1,
                      workers=workers)
    clean_records, records = read_jsonl(out / "clean_ocr.jsonl"), read_jsonl(out / "degraded_ocr.jsonl")
    curves = {s["id"]: corpus_cer([r for r in records if r["severity_id"] == s["id"]])
              for s in config["degradation"]["severities"]}
    report = {"split": "validation", "test_inspected": False,
              "clean_corpus_cer": corpus_cer(clean_records), "severity_corpus_cer": curves,
              "elapsed_seconds": time.perf_counter()-start,
              "engine": clean_records[0]["ocr_provenance"],
              "needs_human_calibration": True,
              "note": "No universal CER cutoffs are imposed. Inspect readable clean text, useful severity variation, and measured runtime; then freeze settings before final testing."}
    write_json(out / "diagnostic.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "config", "tessdata-dir", "output-dir"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--tesseract", default="tesseract")
    parser.add_argument("--workers", type=int,
                        help="Override config ocr.workers (default: config value or 1)")
    args = parser.parse_args()
    print(run_diagnostic(args.data_root, read_json(args.config), args.tessdata_dir,
                         args.output_dir, args.tesseract, workers=args.workers))


if __name__ == "__main__":
    main()
