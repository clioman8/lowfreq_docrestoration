"""Ordered OCR concurrency must preserve scientific outputs and atomic files."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from docrestore import diagnose, evaluate


class ParallelEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tessdata = self.root / "tessdata"
        self.tessdata.mkdir()
        (self.tessdata / "eng.traineddata").write_bytes(b"frozen-model-fixture")
        self.manifest = self.root / "manifest.jsonl"
        self.predictions = self.root / "predictions.jsonl"
        self.rows, predictions = [], []
        pixels = np.arange(20 * 24, dtype=np.uint16).reshape(20, 24) % 256
        for index in range(4):
            for variant, offset in (("clean", 0), ("degraded", 31), ("restored", 7)):
                image = ((pixels + offset) % 256).astype(np.uint8)
                Image.fromarray(image).save(self.root / f"s{index}.{variant}.png")
            self.rows.append({"sample_id": f"s{index}", "severity_id": "S1",
                              "gt_text": f"Caf\u00e9 {index}.", "split": "validation",
                              "clean_path": f"s{index}.clean.png",
                              "degraded_path": f"s{index}.degraded.png"})
            predictions.append({"sample_id": f"s{index}", "severity_id": "S1",
                                "restored_path": f"s{index}.restored.png",
                                "seed": 71, "method": "C"})
        self.write_manifest(self.rows)
        self.predictions.write_text("".join(json.dumps(row) + "\n" for row in predictions),
                                    encoding="utf-8")
        (self.root / "checkpoint_provenance.json").write_text(
            json.dumps({"step": 200, "checkpoint_sha256": "fixture"}), encoding="utf-8")
        self.lock = threading.Lock()
        self.active = self.maximum_active = 0
        self.completed = []
        self.failure = None

    def write_manifest(self, rows):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows),
                                 encoding="utf-8")

    def fake_tesseract(self, command, **kwargs):
        self.assertEqual(kwargs["env"]["OMP_THREAD_LIMIT"], "1")
        self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "1")
        if command[1] == "--version":
            return subprocess.CompletedProcess(command, 0, "fixture-tesseract 1\n", "")
        index = int(Path(command[1]).name[1])
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            # Later rows finish first so serial/parallel byte equivalence proves
            # manifest order, not just identical unordered record contents.
            time.sleep(.04 if index == 0 else .002)
            if index == self.failure:
                raise subprocess.CalledProcessError(9, command, stderr="fixture OCR failure")
            text = f"  Caf{'e' if index == 1 else chr(233)} {index}.\n"
            with self.lock:
                self.completed.append(index)
            return subprocess.CompletedProcess(command, 0, text, "fixture stderr\n")
        finally:
            with self.lock:
                self.active -= 1

    def run_evaluation(self, output, **kwargs):
        with patch("docrestore.evaluate.subprocess.run", side_effect=self.fake_tesseract):
            return evaluate.evaluate_manifest(self.manifest, self.root, output, self.tessdata,
                                               **kwargs)

    def test_parallel_is_byte_identical_for_all_modes_and_provenance(self):
        for mode in ("clean", "degraded", "restored"):
            with self.subTest(mode=mode):
                arguments = {"mode": mode}
                if mode == "restored":
                    arguments.update(predictions_manifest=self.predictions, method="C", seed=71)
                serial = self.root / f"{mode}.serial.jsonl"
                parallel = self.root / f"{mode}.parallel.jsonl"
                serial_summary = self.run_evaluation(serial, **arguments)
                self.maximum_active = 0
                self.completed.clear()
                parallel_summary = self.run_evaluation(parallel, workers=3, **arguments)
                self.assertEqual(serial.read_bytes(), parallel.read_bytes())
                self.assertGreater(self.maximum_active, 1)
                self.assertNotEqual(self.completed, [0, 1, 2, 3])
                serial_summary.pop("output")
                parallel_summary.pop("output")
                self.assertEqual(serial_summary, parallel_summary)
                self.assertEqual(parallel_summary["n_records"], 4)

    def test_failure_keeps_destination_and_original_error_with_no_partial_file(self):
        self.failure = 1
        for workers in (1, 3):
            with self.subTest(workers=workers):
                output = self.root / "failed.jsonl"
                output.write_bytes(b"existing-result\n")
                with self.assertRaisesRegex(RuntimeError, "Tesseract failed.*s1.*fixture OCR failure") as error:
                    self.run_evaluation(output, mode="degraded", workers=workers)
                self.assertIsInstance(error.exception.__cause__, subprocess.CalledProcessError)
                self.assertEqual(output.read_bytes(), b"existing-result\n")
                self.assertEqual(list(self.root.glob("failed.jsonl.*.tmp")), [])

    def test_earlier_ocr_failure_precedes_later_manifest_error(self):
        self.failure = 0
        self.write_manifest([self.rows[0], self.rows[1], self.rows[1]])
        for workers in (1, 3):
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(RuntimeError, "Tesseract failed.*s0"):
                    self.run_evaluation(self.root / "ordered-errors.jsonl", workers=workers)

    def test_input_errors_are_atomic_for_empty_duplicate_and_invalid_json(self):
        invalid_inputs = [
            ("", "Evaluation manifest is empty"),
            (json.dumps(self.rows[0]) + "\n" + json.dumps(self.rows[0]) + "\n",
             "Duplicate input key"),
            (json.dumps(self.rows[0]) + "\n{invalid\n", "Invalid JSON"),
        ]
        for content, message in invalid_inputs:
            for workers in (1, 3):
                with self.subTest(message=message, workers=workers):
                    self.manifest.write_text(content, encoding="utf-8")
                    output = self.root / "invalid.jsonl"
                    output.write_bytes(b"existing-result\n")
                    with self.assertRaisesRegex(ValueError, message):
                        self.run_evaluation(output, workers=workers)
                    self.assertEqual(output.read_bytes(), b"existing-result\n")
                    self.assertEqual(list(self.root.glob("invalid.jsonl.*.tmp")), [])

    def test_workers_must_be_a_positive_integer(self):
        for workers in (0, -1, 1.5, True, "2", None):
            with self.subTest(workers=workers):
                with patch("docrestore.evaluate.subprocess.run") as run:
                    with self.assertRaisesRegex(ValueError, "workers must be a positive integer"):
                        evaluate.evaluate_manifest(self.manifest, self.root, self.root / "bad.jsonl",
                                                   self.tessdata, workers=workers)
                    run.assert_not_called()

    def test_evaluate_cli_forwards_workers_and_default_remains_one(self):
        base = ["--manifest", "input", "--images-root", "images", "--output", "output",
                "--tessdata-dir", "tessdata"]
        for arguments, expected in ((base, 1), (base + ["--workers", "4"], 4)):
            with patch("docrestore.evaluate.evaluate_manifest", return_value={}) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    evaluate.main(arguments)
                self.assertEqual(run.call_args.kwargs["workers"], expected)

    def test_diagnostic_config_and_explicit_override(self):
        config = {"ocr": {"workers": 3}, "degradation": {"severities": [{"id": "S1"}]}}
        eval_manifest = self.root / "eval_manifest.jsonl"
        eval_manifest.write_bytes(self.manifest.read_bytes())
        for override, expected in ((None, 3), (2, 2)):
            output = self.root / f"diagnostic-{expected}"
            captured = []

            def evaluate_and_capture(*args, **kwargs):
                captured.append(kwargs["workers"])
                with patch("docrestore.evaluate.subprocess.run", side_effect=self.fake_tesseract):
                    return evaluate.evaluate_manifest(*args, **kwargs)

            with patch("docrestore.diagnose.evaluate_manifest", side_effect=evaluate_and_capture):
                report = diagnose.run_diagnostic(self.root, config, self.tessdata, output,
                                                 workers=override)
            self.assertEqual(captured, [expected, expected])
            self.assertFalse(report["test_inspected"])


if __name__ == "__main__":
    unittest.main()
