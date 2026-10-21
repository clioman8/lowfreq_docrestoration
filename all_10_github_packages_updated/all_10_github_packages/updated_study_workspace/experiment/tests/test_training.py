"""Small numerical integration checks, independent of OCR assets and datasets."""

import tempfile
import unittest
from argparse import Namespace
import json
from pathlib import Path
import random
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    import docrestore.train as train_module
    from docrestore.model import UNet
    from docrestore.train import (seed_everything, validation_score, fit,
                                  engine_version_matches, atomic_torch_save, evaluation_steps)
    from docrestore.operators import forward_torch
    from docrestore.losses import (build_fourier_basis, data_loss,
                                   full_consistency_loss,
                                   subspace_consistency_loss, ssim_torch)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def _one_step(self, group, device="cpu"):
        seed_everything(17)
        model = UNet((4, 8, 12, 16)).to(device)
        initial = {name: value.detach().cpu().clone()
                   for name, value in model.state_dict().items()}
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        generator = torch.Generator().manual_seed(719)
        clean = torch.rand((2, 1, 32, 40), generator=generator).to(device)
        eta = {"kappa": torch.tensor([.2, .8], device=device),
               "alpha": torch.tensor([.1, .3], device=device), "T": .7}
        with torch.no_grad():
            observed = forward_torch(clean, **eta, n_steps=3)
        pred = model(observed)
        reconstruction = data_loss(pred, clean)
        residual = forward_torch(pred, **eta, n_steps=3) - observed
        full = full_consistency_loss(residual)
        Q = torch.tensor(build_fourier_basis(32, 40, 12), dtype=pred.dtype, device=device)
        sub = subspace_consistency_loss(residual, Q)
        # These are exactly the output-gradient diagnostics recorded by fit().
        for component in (reconstruction, full, sub):
            gradient = torch.autograd.grad(component, pred, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient).all())
        total = reconstruction if group == "A" else reconstruction + .1 * (
            full if group == "B" else sub)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.assertTrue(torch.isfinite(total))
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        optimizer.step()
        changed = any(not torch.equal(value.detach().cpu(), initial[name])
                      for name, value in model.state_dict().items())
        self.assertTrue(changed)
        return model, initial, observed

    def test_float32_model_operator_losses_all_groups(self):
        initial_reference = observed_reference = None
        for group in ("A", "B", "C"):
            model, initial, observed = self._one_step(group)
            if initial_reference is None:
                initial_reference, observed_reference = initial, observed
            else:
                for key, value in initial.items():
                    self.assertTrue(torch.equal(value, initial_reference[key]))
                self.assertTrue(torch.equal(observed, observed_reference))

    def test_validation_selection_is_corpus_weighted_and_excludes_s0(self):
        rows = [{"severity_id": "S0", "gt_chars": 100, "edit_distance": 999},
                {"severity_id": "S1", "gt_chars": 10, "edit_distance": 1},
                {"severity_id": "S1", "gt_chars": 90, "edit_distance": 3},
                {"severity_id": "S2", "gt_chars": 100, "edit_distance": 3}]
        maximum, mean = validation_score(rows, ["S1", "S2"])
        self.assertAlmostEqual(maximum, .04)
        self.assertAlmostEqual(mean, .035)
        # Lexicographic max-first, mean-second selection used in fit().
        self.assertLess((.04, .035), (.05, .02))
        self.assertLess((.04, .035), (.04, .04))
        with self.assertRaises(ValueError):
            validation_score(rows, ["S1", "S3"])

    def test_checkpoint_weights_only_roundtrip(self):
        model, _, observed = self._one_step("C")
        model.eval()
        with torch.no_grad():
            before = model(observed)
        checkpoint = {"model_state": model.state_dict(),
                      "config": {"model": {"channels": [4, 8, 12, 16]}},
                      "metadata": {"group": "C", "seed": 17},
                      "step": 1, "validation_score": [.3, .2]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            torch.save(checkpoint, path)
            restored = torch.load(path, map_location="cpu", weights_only=True)
        replica = UNet((4, 8, 12, 16))
        replica.load_state_dict(restored["model_state"])
        replica.eval()
        with torch.no_grad():
            after = replica(observed)
        torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_engine_version_accepts_official_build_but_requires_exact_core(self):
        for observed in ("tesseract 5.5.3", "tesseract v5.5.3.20260724"):
            self.assertTrue(engine_version_matches(observed, "5.5.3"))
        for observed in ("tesseract 5.5.30", "tesseract v5.3.4", "other 5.5.3"):
            self.assertFalse(engine_version_matches(observed, "5.5.3"))

    def test_explicit_validation_schedule_and_legacy_every(self):
        self.assertEqual(evaluation_steps({"eval_every": 3}, 8), (3, 6, 8))
        self.assertEqual(evaluation_steps({"eval_steps": [1, 3, 8]}, 8), (1, 3, 8))
        self.assertEqual(evaluation_steps({"eval_steps": [250, 500, 1000, 2000, 4000,
                                                       6000, 8000, 10000]}, 10000),
                         (250, 500, 1000, 2000, 4000, 6000, 8000, 10000))
        for invalid in ([], [0, 8], [-1, 8], [3, 3, 8], [3, 1, 8], [1, 9],
                        [True, 8], [1.5, 8], [1, 3], "1,8"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evaluation_steps({"eval_steps": invalid}, 8)

    def test_atomic_checkpoint_failure_preserves_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.pt"
            atomic_torch_save({"step": 2}, path)
            with mock.patch.object(train_module.torch, "save", side_effect=OSError("interrupted save")):
                with self.assertRaises(OSError):
                    atomic_torch_save({"step": 4}, path)
            self.assertEqual(torch.load(path, weights_only=True), {"step": 2})
            self.assertEqual(list(Path(temporary).iterdir()), [path])

    def test_resume_matches_uninterrupted_updates_across_loader_epoch(self):
        """Replay from step 2 after work at steps 3/4, including all RNG streams."""
        from PIL import Image
        from docrestore.common import read_json, read_jsonl, write_json, write_jsonl
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, tessdata = root / "data", root / "tessdata"
            data.mkdir()
            tessdata.mkdir()
            (tessdata / "eng.traineddata").write_bytes(b"mock OCR asset")
            generator = np.random.default_rng(901)
            rows = []
            # Five items with batch=2 gives an uneven third batch and an epoch
            # boundary at step 4, exercising lazy RandomSampler replay.
            for index in range(5):
                path = data / f"train-{index}.png"
                Image.fromarray(generator.integers(0, 256, (16, 16), dtype=np.uint8)).save(path)
                rows.append({"sample_id": str(index), "split": "train", "clean_path": path.name})
            clean = generator.integers(0, 256, (16, 16), dtype=np.uint8)
            Image.fromarray(clean).save(data / "validation.png")
            np.save(data / "validation.npy", clean.astype(np.float32) / 255)
            eta = {"kappa": .4, "alpha": .04, "T": 1., "n_steps": 3, "h": 1.}
            val_rows = [{"sample_id": "validation", "split": "validation", "severity_id": "S1",
                         "clean_path": "validation.png", "degraded_path": "validation.png",
                         "degraded_array_path": "validation.npy", "gt_text": "fixture", "eta": eta}]
            write_jsonl(data / "manifest.jsonl", rows)
            write_jsonl(data / "eval_manifest.jsonl", val_rows)
            write_json(data / "dataset_metadata.json", {"settings": {}})
            config = {"degradation": {"n_steps": 3, "h": 1.,
                                      "train": {"kappa": [.1, .7], "alpha": [0, .1], "T": 1.},
                                      "severities": [{"id": "S1", "kappa": .4, "alpha": .04, "T": 1.}]},
                      "model": {"channels": [4, 8, 12, 16]},
                      "loss": {"beta": .1, "subspace_dim": 8},
                      "training": {"batch_size": 2, "learning_rate": .0002, "max_steps": 6,
                                   "eval_every": 2, "diagnostics_every": 2, "cpu_threads": 1},
                      "analysis": {"shifted_severities": ["S1"]}}
            config_path = root / "config.json"
            write_json(config_path, config)

            def arguments(run, resume=False):
                return Namespace(config=str(config_path), data_root=str(data), run_dir=str(run),
                                 tessdata_dir=str(tessdata), group="C", seed=17, lambda_value=.1,
                                 stage="pilot", device="cpu", tesseract="mock-tesseract", resume=resume)

            def fake_ocr(manifest, data_root, output, *unused):
                records = read_jsonl(manifest)
                for record in records:
                    image = np.asarray(Image.open(record["restored_path"]), dtype=float)
                    # Controlled global randomness makes omitted RNG restoration
                    # observable even though the production OCR is deterministic.
                    jitter = random.random() + np.random.random() + torch.rand(1).item()
                    record.update(gt_chars=1000,
                                  edit_distance=int(np.abs(image - clean).mean() + 10 * jitter),
                                  evaluated_path=record["restored_path"])
                write_jsonl(output, records)
                return records

            uninterrupted, resumed = root / "uninterrupted", root / "resumed"
            shared = [mock.patch.object(train_module, "check_eng", return_value={"mock": True}),
                      mock.patch.object(train_module.subprocess, "run", return_value=SimpleNamespace(
                          stdout="tesseract v5.5.3.20260724\n"))]
            with shared[0], shared[1]:
                with mock.patch.object(train_module, "evaluate_predictions", side_effect=fake_ocr):
                    fit(arguments(uninterrupted))
                calls = 0

                def interrupt_second_validation(*args):
                    nonlocal calls
                    calls += 1
                    result = fake_ocr(*args)
                    if calls == 2:
                        raise RuntimeError("simulated interruption after unsaved updates")
                    return result

                with mock.patch.object(train_module, "evaluate_predictions", side_effect=interrupt_second_validation):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        fit(arguments(resumed))
                self.assertEqual(torch.load(resumed / "last.pt", weights_only=True)["step"], 2)
                self.assertEqual(read_jsonl(resumed / "training_log.jsonl")[-1]["step"], 3)
                with mock.patch.object(train_module, "evaluate_predictions", side_effect=fake_ocr):
                    fit(arguments(resumed, resume=True))
                expected = torch.load(uninterrupted / "last.pt", weights_only=True)
                actual = torch.load(resumed / "last.pt", weights_only=True)

                def assert_nested_equal(first, second):
                    if isinstance(first, torch.Tensor):
                        torch.testing.assert_close(first, second, rtol=0, atol=0)
                    elif isinstance(first, dict):
                        self.assertEqual(first.keys(), second.keys())
                        for key in first:
                            assert_nested_equal(first[key], second[key])
                    elif isinstance(first, (tuple, list)):
                        self.assertEqual(len(first), len(second))
                        for left, right in zip(first, second):
                            assert_nested_equal(left, right)
                    else:
                        self.assertEqual(first, second)

                assert_nested_equal(expected["model_state"], actual["model_state"])
                for key in ("optimizer_state", "rng_state", "degradation_generator_state",
                            "loader_epoch_start_state", "batches_in_epoch", "best_score", "best_step"):
                    assert_nested_equal(expected["resume_state"][key], actual["resume_state"][key])
                self.assertEqual(read_jsonl(uninterrupted / "training_log.jsonl"),
                                 read_jsonl(resumed / "training_log.jsonl"))
                self.assertEqual(read_json(resumed / "summary.json")["resumed_from_step"], 2)
                # Keeping max_steps in the hash prevents changing a completed
                # run into a different training-budget experiment on resume.
                config["training"]["max_steps"] = 8
                write_json(config_path, config)
                with self.assertRaisesRegex(ValueError, "Resume config differs"):
                    fit(arguments(resumed, resume=True))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),
                         "CUDA hardware unavailable; CPU checks do not certify GPU execution")
    def test_strict_deterministic_cuda_backward(self):
        self._one_step("C", "cuda")


if __name__ == "__main__":
    unittest.main()
