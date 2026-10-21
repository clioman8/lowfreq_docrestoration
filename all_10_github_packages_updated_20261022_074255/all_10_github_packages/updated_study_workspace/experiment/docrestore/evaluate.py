"""Run one frozen Tesseract pipeline on a manifest; never train or correct OCR."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import math
import os
from numbers import Integral
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image

from .metrics import NORMALIZATION, character_error_rate, psnr, ssim


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_image(root: Path, path: str) -> Path:
    result = Path(path)
    return result if result.is_absolute() else root / result


def finite_json(value):
    # JSON has no infinity literal. Preserve perfect-image PSNR explicitly.
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value == math.inf else "-inf" if value == -math.inf else None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(item) for item in value]
    return value


def _ordered_map(function, records, workers):
    """Bound in-flight work while yielding results/errors in input order."""
    if workers == 1:
        for record in records:
            yield function(record)
        return
    iterator = iter(records)
    pending = deque()
    exhausted = False
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ocr")

    def submit_next():
        nonlocal exhausted
        try:
            record = next(iterator)
        except StopIteration:
            exhausted = True
        except Exception as exc:
            # A later manifest error must not hide an earlier OCR failure.
            failure = Future()
            failure.set_exception(exc)
            pending.append(failure)
            exhausted = True
        else:
            pending.append(executor.submit(function, record))

    try:
        # Do not materialize one Future for every record in a large manifest.
        for _ in range(2 * workers):
            if exhausted:
                break
            submit_next()
        while pending:
            future = pending.popleft()
            yield future.result()
            if not exhausted:
                submit_next()
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def evaluate_manifest(manifest, images_root, output, tessdata_dir, *,
                      tesseract="tesseract", mode="clean", image_field=None,
                      predictions_manifest=None, method=None, seed=None,
                      psm=6, oem=1, timeout=120, lang="eng", run_provenance=None,
                      workers=1) -> dict:
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    if oem != 1:
        raise ValueError("This frozen tessdata_best protocol requires OEM 1")
    root, destination = Path(images_root), Path(output)
    tessdata = Path(tessdata_dir).resolve()
    language_data = tessdata / f"{lang}.traineddata"
    if not language_data.is_file():
        raise FileNotFoundError(language_data)
    env = dict(os.environ, OMP_THREAD_LIMIT="1", OMP_NUM_THREADS="1")
    engine_version = subprocess.run([str(tesseract), "--version"], env=env,
                                    check=True, capture_output=True, text=True,
                                    timeout=30).stdout.strip()
    provenance = {"tesseract_version": engine_version,
                  "tessdata_sha256": sha256_file(language_data),
                  "language": lang, "oem": oem, "psm": psm,
                  "normalization": NORMALIZATION,
                  "preprocessing": "none_external_tesseract_internal_only",
                  "omp_thread_limit": 1,
                  "ssim_definition": "gaussian11_sigma1.5_population_valid_mean_clamp_range1_v3"}
    input_provenance = {"evaluation_manifest_sha256": sha256_file(Path(manifest)),
                        "predictions_manifest_sha256": sha256_file(Path(predictions_manifest))
                        if predictions_manifest else None}
    restoration_provenance = None
    provenance_path = Path(run_provenance) if run_provenance else (
        Path(predictions_manifest or manifest).resolve().parent / "checkpoint_provenance.json")
    if mode == "restored" and (run_provenance or provenance_path.is_file()):
        restoration_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        restoration_provenance = {**restoration_provenance,
                                  "provenance_file_sha256": sha256_file(provenance_path)}
    predictions = {}
    if predictions_manifest:
        for item in read_jsonl(predictions_manifest):
            key = (str(item["sample_id"]), str(item.get("severity_id", "clean")))
            if key in predictions:
                raise ValueError(f"Duplicate prediction key {key}")
            predictions[key] = item
    image_field = image_field or {"clean": "clean_path", "degraded": "degraded_path",
                                  "restored": "restored_path"}[mode]
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = total_errors = total_chars = 0
    temporary = None
    seen = set()

    def prepared_records():
        # Validation and prediction joins remain in manifest order.
        for source in read_jsonl(manifest):
            item = dict(source)
            key = (str(item["sample_id"]), str(item.get("severity_id", "clean")))
            if key in seen:
                raise ValueError(f"Duplicate input key {key}")
            seen.add(key)
            if predictions_manifest:
                if key not in predictions:
                    raise ValueError(f"No prediction for {key}")
                item.update(predictions[key])
            yield item

    def evaluate_record(item):
        key = (str(item["sample_id"]), str(item.get("severity_id", "clean")))
        if image_field not in item:
            raise ValueError(f"Sample {key} lacks {image_field}")
        image_path = resolve_image(root, item[image_field]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        command = [str(tesseract), str(image_path), "stdout", "-l", lang,
                   "--oem", str(oem), "--psm", str(psm),
                   "--tessdata-dir", str(tessdata)]
        try:
            process = subprocess.run(command, env=env, capture_output=True,
                                     text=True, encoding="utf-8", timeout=timeout,
                                     check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Tesseract failed on {key}: {exc.stderr}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Tesseract timed out on {key}") from exc
        text_scores = character_error_rate(item["gt_text"], process.stdout)
        clean_path = resolve_image(root, item["clean_path"]).resolve()
        with Image.open(clean_path) as clean_image:
            clean = np.asarray(clean_image.convert("L"), dtype=np.float64) / 255
        with Image.open(image_path) as evaluated_image:
            evaluated = np.asarray(evaluated_image.convert("L"), dtype=np.float64) / 255
        return {**item, **text_scores, "ocr_text": process.stdout,
                "ocr_stderr": process.stderr,
                "mode": mode, "method": method or item.get("method", mode),
                "seed": (seed if seed is not None else item.get("seed"))
                if mode == "restored" else None,
                "run_kind": "training_seed" if mode == "restored" else "nonlearned_reference",
                "evaluated_path": str(image_path), "image_field": image_field,
                "psnr_db": psnr(clean, evaluated), "ssim": ssim(clean, evaluated),
                "ocr_provenance": provenance,
                "evaluation_provenance": input_provenance,
                "restoration_provenance": restoration_provenance}

    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                         dir=destination.parent,
                                         prefix=destination.name + ".", suffix=".tmp") as stream:
            temporary = Path(stream.name)
            for result in _ordered_map(evaluate_record, prepared_records(), workers):
                stream.write(json.dumps(finite_json(result), ensure_ascii=False,
                                        allow_nan=False) + "\n")
                count += 1
                total_errors += int(result["edit_distance"])
                total_chars += int(result["gt_chars"])
            if not count:
                raise ValueError("Evaluation manifest is empty")
        temporary.replace(destination)
        return {"output": str(destination), "n_records": count,
                "corpus_cer": total_errors / total_chars,
                "note": "Pooled across all input records; aggregate separately by severity for reporting"}
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tessdata-dir", required=True)
    parser.add_argument("--tesseract", default="tesseract")
    parser.add_argument("--mode", choices=["clean", "degraded", "restored"], default="clean")
    parser.add_argument("--image-field")
    parser.add_argument("--predictions-manifest")
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--psm", type=int, default=6)
    parser.add_argument("--oem", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--lang", default="eng")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent single-threaded Tesseract processes (default: 1)")
    parser.add_argument("--run-provenance", help="Optional checkpoint/run provenance JSON; otherwise auto-detect beside manifest")
    args = vars(parser.parse_args(argv))
    print(json.dumps(evaluate_manifest(**args), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
