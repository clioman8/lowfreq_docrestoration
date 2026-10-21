"""Matched-seed U-Net training and restoration export.

Checkpoint selection uses validation OCR only: maximum shifted corpus CER,
then mean shifted CER. Test images never participate in fit().
The restoration network receives only y, never eta. Online training pairs
stay floating point; saved evaluation inputs/outputs are explicitly 8-bit PNG.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from .common import read_json, read_jsonl, write_json, write_jsonl, resolve_path, config_hash, sha256_file
from .model import UNet
from .operators import forward_torch
from .losses import build_fourier_basis, data_loss, full_consistency_loss, subspace_consistency_loss
from .assets import check_eng


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def engine_version_matches(observed, expected):
    """Compare the exact three-part version, allowing official v/build suffixes."""
    actual = re.match(r"^tesseract\s+v?(\d+)\.(\d+)\.(\d+)"
                      r"(?:[.\-+][0-9A-Za-z_.+\-]+)?(?:\s|$)", observed.strip(), re.I)
    wanted = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", str(expected).strip())
    return actual is not None and wanted is not None and actual.groups() == wanted.groups()


def evaluation_steps(training, max_steps):
    """Explicit schedules must be sorted, unique and include the final step.

    Without eval_steps, preserve eval_every checkpoints plus max_steps.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if "eval_steps" in training:
        steps = training["eval_steps"]
        if (not isinstance(steps, list) or not steps
                or any(isinstance(step, bool) or not isinstance(step, int)
                       or step < 1 or step > max_steps for step in steps)
                or steps != sorted(set(steps))):
            raise ValueError("eval_steps must be a nonempty, sorted list of unique positive integers <= max_steps")
        if steps[-1] != max_steps:
            raise ValueError("eval_steps must explicitly include max_steps")
        return tuple(steps)
    every = int(training.get("eval_every", 100))
    if every < 1:
        raise ValueError("eval_every must be positive")
    return tuple(sorted(set(range(every, max_steps + 1, every)) | {max_steps}))


def atomic_torch_save(value, destination):
    """Publish a complete checkpoint without replacing a previous one on failure."""
    destination = Path(destination)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        torch.save(value, temporary)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def capture_rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"name": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": numpy_state[2], "has_gauss": numpy_state[3],
                      "cached_gaussian": numpy_state[4]},
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value["name"], np.asarray(value["keys"], dtype=np.uint32),
                         value["position"], value["has_gauss"], value["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the original CUDA device count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def restore_training_log(path, saved_step, saved_entry):
    """Discard work after the snapshot and repair a last, incompletely written line."""
    path = Path(path)
    retained = []
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            break
        step = entry.get("step")
        if not isinstance(step, int) or step > saved_step:
            break
        if step != len(retained) + 1:
            raise ValueError("Training log has missing or duplicate steps before resume checkpoint")
        retained.append(entry)
    if len(retained) == saved_step - 1 and saved_entry.get("step") == saved_step:
        retained.append(saved_entry)
    if len(retained) != saved_step:
        raise ValueError("Training log is incomplete before the resume checkpoint")
    temporary = path.with_name(path.name + ".resume.tmp")
    write_jsonl(temporary, retained)
    temporary.replace(path)


def image_tensor(path):
    if Path(path).suffix == ".npy":
        value = np.load(path, allow_pickle=False).astype(np.float32)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError("Expected a finite, two-dimensional observation array")
    else:
        with Image.open(path) as image:
            value = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(value.copy())[None]


class CleanDataset(Dataset):
    def __init__(self, data_root):
        self.root = Path(data_root)
        self.rows = [r for r in read_jsonl(self.root / "manifest.jsonl") if r["split"] == "train"]
        if not self.rows:
            raise ValueError("Training manifest has no train records")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return image_tensor(resolve_path(self.root, self.rows[index]["clean_path"]))


def validation_score(rows, shifted_ids):
    values = []
    for severity in shifted_ids:
        subset = [r for r in rows if r["severity_id"] == severity]
        if not subset:
            raise ValueError(f"Validation output missing severity {severity}")
        chars = sum(int(r["gt_chars"]) for r in subset)
        if chars <= 0:
            raise ValueError("Empty validation transcription")
        values.append(sum(int(r["edit_distance"]) for r in subset) / chars)
    return max(values), float(np.mean(values))


def export_predictions(model, rows, data_root, output_dir, device, method, seed):
    """Export uint8 restoration images and their complete correspondence manifest."""
    output_dir = Path(output_dir).resolve()
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    exported = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for row in rows:
            input_field = "degraded_array_path" if row.get("degraded_array_path") else "degraded_path"
            image = image_tensor(resolve_path(data_root, row[input_field]))[None].to(device)
            restored = model(image)[0, 0].cpu().numpy()
            image_path = output_dir / "images" / f"{row['sample_id']}__{row['severity_id']}.png"
            Image.fromarray(np.rint(np.clip(restored, 0, 1) * 255).astype(np.uint8)).save(image_path)
            item = dict(row)
            item.update(restored_path=str(image_path), method=method, seed=seed)
            exported.append(item)
    model.train(was_training)
    manifest = output_dir / "predictions_manifest.jsonl"
    write_jsonl(manifest, exported)
    return manifest


def evaluate_predictions(manifest, data_root, output, tessdata_dir, tesseract, config, method, seed):
    ocr = config.get("ocr", {})
    command = [sys.executable, "-m", "docrestore.evaluate", "--manifest", str(manifest),
               "--images-root", str(data_root), "--output", str(output),
               "--image-field", "restored_path", "--mode", "restored",
               "--tessdata-dir", str(tessdata_dir), "--tesseract", tesseract,
               "--psm", str(ocr.get("psm", 6)), "--oem", "1", "--method", method, "--seed", str(seed),
               "--workers", str(ocr.get("workers", 1))]
    subprocess.run(command, check=True)
    return read_jsonl(output)


def fit(args):
    config = read_json(args.config)
    if args.stage == "final" and not config.get("configuration_frozen", False):
        raise ValueError("Final training requires configuration_frozen=true after a successful pilot")
    seed_everything(args.seed)
    torch.set_num_threads(int(config.get("training", {}).get("cpu_threads", 2)))
    device = select_device(args.device)
    root, run = Path(args.data_root).resolve(), Path(args.run_dir).resolve()
    resume = bool(getattr(args, "resume", False))
    if (run / "run_metadata.json").exists() and not resume:
        raise FileExistsError("Run directory already contains a run; choose a new directory")
    if resume and not (run / "run_metadata.json").is_file():
        raise FileNotFoundError("--resume requires an existing run_metadata.json and last.pt")
    run.mkdir(parents=True, exist_ok=True)
    tessdata = Path(args.tessdata_dir).resolve()
    if not (tessdata / "eng.traineddata").is_file():
        raise FileNotFoundError("Place the fixed eng.traineddata in --tessdata-dir")
    eng_verification = check_eng(tessdata / "eng.traineddata")
    version = subprocess.run([args.tesseract, "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    expected = config.get("ocr", {}).get("expected_engine_version", "5.5.3")
    if args.stage == "final" and not engine_version_matches(version, expected):
        raise ValueError(f"Final OCR engine must be {expected}; observed {version}")
    actual_data_meta = read_json(root / "dataset_metadata.json")
    if args.stage == "final" and not actual_data_meta.get("settings", {}).get("language_validation", False):
        raise ValueError("Final training needs a verified English book allowlist and data.language_validation=true")
    if args.stage == "final" and not actual_data_meta.get("settings", {}).get("source_split_manifest"):
        raise ValueError("Final data must reuse the prespecified source split via data.source_split_manifest")
    configured_degradation = config["degradation"]
    val_rows = [r for r in read_jsonl(root / "eval_manifest.jsonl") if r["split"] == "validation"]
    if not val_rows or any(r["split"] == "test" for r in val_rows):
        raise ValueError("A nonempty, separate validation set is required")
    severity_specs = {s["id"]: s for s in configured_degradation["severities"]}
    for row in val_rows:
        spec = severity_specs.get(row["severity_id"])
        expected_eta = ({k: float(spec[k]) for k in ("kappa", "alpha", "T")} if spec else {})
        expected_eta.update(n_steps=int(configured_degradation["n_steps"]), h=float(configured_degradation["h"]))
        if spec is None or row.get("eta") != expected_eta:
            raise ValueError("Stored validation degradation differs from the training configuration")
        if not row.get("degraded_array_path"):
            raise ValueError("Regenerate evaluation data with floating-point observations; PNG is OCR export only")
    train_config = config.get("training", {})
    dataset = CleanDataset(root)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=int(train_config.get("batch_size", 8)), shuffle=True,
                        num_workers=0, generator=loader_generator, drop_last=False)
    model = UNet(tuple(config.get("model", {}).get("channels", [32, 64, 128, 256]))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config.get("learning_rate", 2e-4)))
    first_shape = dataset[0].shape[-2:]
    basis_np = build_fourier_basis(*first_shape, int(config.get("loss", {}).get("subspace_dim", 32)))
    basis = torch.as_tensor(basis_np, dtype=torch.float32, device=device)
    orthogonality_error = float(np.max(np.abs(basis_np.T @ basis_np - np.eye(basis_np.shape[1]))))
    if orthogonality_error > 1e-9:
        raise ValueError("Fourier test basis failed orthogonality verification")
    degradation_generator = torch.Generator().manual_seed(args.seed + 100003)
    regularization = 0.0 if args.group == "A" else (0.1 if args.lambda_value is None else args.lambda_value)
    if regularization < 0 or (args.group == "A" and args.lambda_value not in (None, 0.0)):
        raise ValueError("lambda must be nonnegative and Group A must use zero")
    max_steps = int(train_config.get("max_steps", 200))
    eval_steps = set(evaluation_steps(train_config, max_steps))
    metadata = {"stage": args.stage, "group": args.group, "seed": args.seed, "lambda": regularization,
                "config": config, "config_sha256": config_hash(config),
                "manifest_sha256": sha256_file(root / "manifest.jsonl"),
                "eval_manifest_sha256": sha256_file(root / "eval_manifest.jsonl"),
                "torch_version": str(torch.__version__), "device": str(device),
                "tesseract_version": version, "eng_sha256": sha256_file(tessdata / "eng.traineddata"),
                "eng_verification": eng_verification,
                "basis_orthogonality_max_error": orthogonality_error,
                "basis_convention": "real-Fourier radial/DC-first/cos-before-sin/exact-m; final pair may be incomplete",
                "parameters": sum(p.numel() for p in model.parameters()),
                "upsampling": "nearest", "ssim_definition": "gaussian11_sigma1.5_population_valid_mean_range1",
                "generation_metadata": actual_data_meta, "prediction_export": "round to uint8 PNG",
                "note": "pilot outputs are engineering checks, not final thesis results" if args.stage == "pilot" else "frozen configuration"}
    shifted = config.get("analysis", {}).get("shifted_severities", ["S1", "S2", "S3", "S4"])
    beta = float(config.get("loss", {}).get("beta", 0.1))
    best, best_step = (float("inf"), float("inf")), None
    first_step, prior_seconds, segments = 1, 0.0, 1
    validation_seconds = 0.0
    loader_epoch_start_state = loader_generator.get_state()
    batches_in_epoch = 0
    if resume:
        checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        state = checkpoint.get("resume_state")
        if not state or state.get("schema_version") != 1:
            raise ValueError("Checkpoint predates exact resume support; start a separate run")
        if config_hash(checkpoint["config"]) != config_hash(config):
            raise ValueError("Resume config differs, including max_steps; the full configuration must match")
        old_metadata = read_json(run / "run_metadata.json")
        for key in ("stage", "group", "seed", "lambda", "config_sha256", "manifest_sha256",
                    "eval_manifest_sha256", "torch_version", "device", "tesseract_version", "eng_sha256"):
            if metadata.get(key) != checkpoint["metadata"].get(key) or old_metadata.get(key) != metadata.get(key):
                raise ValueError(f"Resume provenance differs for {key}")
        if state["cuda_device_count"] != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
            raise ValueError("Resume requires the original CUDA device count")
        saved_step = int(checkpoint["step"])
        if not 0 < saved_step <= max_steps:
            raise ValueError("Invalid resume checkpoint step")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        degradation_generator.set_state(state["degradation_generator_state"].cpu())
        loader_epoch_start_state = state["loader_epoch_start_state"].cpu()
        loader_generator.set_state(loader_epoch_start_state)
        batches_in_epoch = int(state["batches_in_epoch"])
        if not 0 < batches_in_epoch <= len(loader):
            raise ValueError("Invalid resume loader position")
        iterator = iter(loader)
        for _ in range(batches_in_epoch):
            next(iterator)
        # Model construction and loader replay must not alter the saved global streams.
        restore_rng_state(state["rng_state"])
        best, best_step = tuple(state["best_score"]), state["best_step"]
        first_step = saved_step + 1
        prior_seconds, validation_seconds = state["elapsed_seconds"], state["validation_seconds"]
        segments = int(state.get("segments", 1)) + 1
        restore_training_log(run / "training_log.jsonl", saved_step, state["log_entry"])
    else:
        write_json(run / "run_metadata.json", metadata)
        iterator = iter(loader)
    start = time.perf_counter()
    with (run / "training_log.jsonl").open("a" if resume else "w", encoding="utf-8") as log:
        for step in range(first_step, max_steps + 1):
            try:
                clean = next(iterator)
            except StopIteration:
                loader_epoch_start_state = loader_generator.get_state()
                iterator = iter(loader)
                batches_in_epoch = 0
                clean = next(iterator)
            batches_in_epoch += 1
            clean = clean.to(device)
            if tuple(clean.shape[-2:]) != tuple(first_shape):
                raise ValueError("All image geometries must agree")
            batch = clean.shape[0]
            eta = {}
            for name in ("kappa", "alpha"):
                lo, hi = map(float, configured_degradation["train"][name])
                if not 0 <= lo <= hi:
                    raise ValueError("Invalid training parameter range")
                eta[name] = (lo + (hi-lo) * torch.rand(batch, generator=degradation_generator)).to(device)
            eta["T"] = float(configured_degradation["train"].get("T", 1))
            with torch.no_grad():
                observed = forward_torch(clean, **eta, n_steps=configured_degradation["n_steps"], h=configured_degradation["h"])
            pred = model(observed)
            data = data_loss(pred, clean, beta=beta)
            residual = forward_torch(pred, **eta, n_steps=configured_degradation["n_steps"], h=configured_degradation["h"]) - observed
            full, sub = full_consistency_loss(residual), subspace_consistency_loss(residual, basis)
            chosen = full if args.group == "B" else sub
            total = data if args.group == "A" else data + regularization * chosen
            entry = {"step": step, "data_loss": data.item(), "full_loss": full.item(),
                     "sub_loss": sub.item(), "total_loss": total.item(),
                     "projected_energy_fraction": sub.item()/max(full.item(), 1e-30)}
            diagnostics_every = int(train_config.get("diagnostics_every", 50))
            if step == 1 or (diagnostics_every > 0 and step % diagnostics_every == 0):
                grad_data = torch.autograd.grad(data, pred, retain_graph=True)[0].norm()
                grad_full = torch.autograd.grad(full, pred, retain_graph=True)[0].norm()
                grad_sub = torch.autograd.grad(sub, pred, retain_graph=True)[0].norm()
                entry.update(output_gradient_data=grad_data.item(), output_gradient_full=grad_full.item(),
                             output_gradient_sub=grad_sub.item(),
                             weighted_to_data_gradient_ratio=(0.0 if args.group == "A" else
                                 (regularization * (grad_full if args.group == "B" else grad_sub)/grad_data.clamp_min(1e-30)).item()))
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            if step in eval_steps:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                val_start = time.perf_counter()
                manifest = export_predictions(model, val_rows, root, run / "validation_current", device, args.group, args.seed)
                val_output = run / "validation_current" / "ocr.jsonl"
                val_result = evaluate_predictions(manifest, root, val_output, tessdata, args.tesseract, config, args.group, args.seed)
                score = validation_score(val_result, shifted)
                validation_seconds += time.perf_counter() - val_start
                entry.update(validation_max_shifted_cer=score[0], validation_mean_shifted_cer=score[1])
                improved = score < best
                if improved:
                    best, best_step = score, step
                checkpoint = {"model_state": model.state_dict(), "config": config,
                              "metadata": metadata, "step": step, "validation_score": list(score),
                              "resume_state": {
                                  "schema_version": 1, "optimizer_state": optimizer.state_dict(),
                                  "rng_state": capture_rng_state(),
                                  "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                                  "degradation_generator_state": degradation_generator.get_state(),
                                  "loader_epoch_start_state": loader_epoch_start_state,
                                  "batches_in_epoch": batches_in_epoch,
                                  "best_score": list(best), "best_step": best_step,
                                  "elapsed_seconds": prior_seconds + time.perf_counter() - start,
                                  "validation_seconds": validation_seconds, "segments": segments,
                                  "log_entry": entry}}
                if improved:
                    atomic_torch_save(checkpoint, run / "best.pt")
                    best_dir = run / "validation_best"
                    shutil.copytree(run / "validation_current", best_dir, dirs_exist_ok=True)
                    retained = []
                    for record in val_result:
                        record = dict(record)
                        for field in ("restored_path", "evaluated_path"):
                            if record.get(field):
                                relative = Path(record[field]).relative_to(run / "validation_current")
                                record[field] = str(best_dir / relative)
                        retained.append(record)
                    write_jsonl(run / "best_validation_ocr.jsonl", retained)
                    write_jsonl(best_dir / "ocr.jsonl", retained)
                    write_jsonl(best_dir / "predictions_manifest.jsonl", retained)
                # Publish last only after the corresponding best images have been retained.
                atomic_torch_save(checkpoint, run / "last.pt")
                print(json.dumps({"group": args.group, "seed": args.seed, **entry}), flush=True)
            log.write(json.dumps(entry) + "\n")
            log.flush()
    total_seconds = prior_seconds + time.perf_counter() - start
    summary = {"best_step": best_step, "best_validation_max_shifted_cer": best[0],
               "best_validation_mean_shifted_cer": best[1], "steps": max_steps,
               "total_seconds": total_seconds, "validation_seconds": validation_seconds,
               "training_and_diagnostics_seconds": total_seconds-validation_seconds,
               "group": args.group, "seed": args.seed, "lambda": regularization, "stage": args.stage}
    if resume:
        summary.update(resumed_from_step=first_step - 1, training_segments=segments,
                       timing_note="Sum of completed snapshot intervals and this segment; abandoned work and resume setup excluded")
    write_json(run / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def predict(args):
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    seed_everything(int(checkpoint["metadata"]["seed"]))
    config = checkpoint["config"]
    torch.set_num_threads(int(config.get("training", {}).get("cpu_threads", 2)))
    model = UNet(tuple(config.get("model", {}).get("channels", [32, 64, 128, 256]))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    rows = read_jsonl(args.manifest)
    if args.split:
        rows = [r for r in rows if r["split"] == args.split]
    if not rows:
        raise ValueError("No prediction records")
    metadata = checkpoint["metadata"]
    out = export_predictions(model, rows, args.data_root, args.output_dir, device, metadata["group"], metadata["seed"])
    write_json(Path(args.output_dir) / "checkpoint_provenance.json", {
        "checkpoint_sha256": sha256_file(args.checkpoint), "training_metadata": metadata,
        "selection_step": checkpoint["step"], "selection_validation_score": checkpoint["validation_score"]})
    print(str(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fit")
    for name in ("config", "data-root", "run-dir", "tessdata-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--group", choices=["A", "B", "C"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lambda-value", type=float)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--tesseract", default="tesseract")
    p.add_argument("--stage", choices=["pilot", "final"], default="pilot")
    p.add_argument("--resume", action="store_true",
                   help="Exactly resume this run's last.pt; full configuration and provenance must match")
    p.set_defaults(func=fit)
    p = sub.add_parser("predict")
    for name in ("checkpoint", "manifest", "data-root", "output-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--split", choices=["validation", "test"])
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.set_defaults(func=predict)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
