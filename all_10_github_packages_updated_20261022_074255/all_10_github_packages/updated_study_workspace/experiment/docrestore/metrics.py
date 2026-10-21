"""Metrics shared by OCR evaluation and result aggregation.

CER normalization is NFC plus collapsed whitespace. Case and punctuation are
preserved. CER is unbounded and pooled by ground-truth character counts.
"""
from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Mapping

import numpy as np
from scipy.ndimage import correlate

NORMALIZATION = "unicode_nfc_whitespace_collapse_case_punctuation_preserved_v1"


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def edit_counts(reference: str, hypothesis: str) -> dict[str, int]:
    """Character Levenshtein counts; ties prefer diagonal, deletion, insertion.

    Inputs are used as supplied; normalize_text must be applied consistently by
    the caller. Two DP rows avoid the memory cost of a full traceback matrix.
    """
    # Each cell is (distance, substitutions, deletions, insertions).
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref_char in enumerate(reference, start=1):
        current = [(i, 0, i, 0)]
        for j, hyp_char in enumerate(hypothesis, start=1):
            diagonal = previous[j - 1]
            substitution = int(ref_char != hyp_char)
            best = (diagonal[0] + substitution,
                    diagonal[1] + substitution, diagonal[2], diagonal[3])
            above = previous[j]
            if above[0] + 1 < best[0]:
                best = (above[0] + 1, above[1], above[2] + 1, above[3])
            left = current[j - 1]
            if left[0] + 1 < best[0]:
                best = (left[0] + 1, left[1], left[2], left[3] + 1)
            current.append(best)
        previous = current
    distance, substitutions, deletions, insertions = previous[-1]
    return {"edit_distance": distance, "substitutions": substitutions,
            "deletions": deletions, "insertions": insertions}


def character_error_rate(reference: str, hypothesis: str) -> dict[str, float | int | str]:
    reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    if not reference:
        raise ValueError("CER requires nonempty normalized ground truth")
    counts = edit_counts(reference, hypothesis)
    return {**counts, "gt_chars": len(reference), "hyp_chars": len(hypothesis),
            "cer": counts["edit_distance"] / len(reference),
            "gt_normalized": reference, "ocr_normalized": hypothesis}


def corpus_cer(records: Iterable[Mapping]) -> float:
    errors = characters = 0
    for record in records:
        n_chars = int(record["gt_chars"])
        distance = int(record["edit_distance"])
        if n_chars <= 0 or distance < 0:
            raise ValueError("Invalid ground-truth character count or edit distance")
        errors += distance
        characters += n_chars
    if not characters:
        raise ValueError("Cannot compute corpus CER for an empty corpus")
    return errors / characters


def psnr(reference: np.ndarray, prediction: np.ndarray, data_range: float = 1.0) -> float:
    reference, prediction = _image_pair(reference, prediction)
    mse = float(np.mean(np.square(reference - prediction)))
    return math.inf if mse == 0 else 10 * math.log10(data_range ** 2 / mse)


def ssim(reference: np.ndarray, prediction: np.ndarray, data_range: float = 1.0) -> float:
    """Grayscale SSIM: Gaussian 11x11, sigma=1.5, population covariance.

    Matches losses.ssim_numpy/ssim_torch: only valid windows are averaged,
    with no padded-border contribution; score clamped to [-1,1]. Tiny QA
    images use the largest odd window fitting the image.
    Constants K1=.01, K2=.03; inputs are grayscale in [0,1]. This NumPy-only
    implementation deliberately avoids importing PyTorch in OCR subprocesses.
    """
    x, y = _image_pair(reference, prediction)
    if data_range <= 0 or not math.isfinite(data_range):
        raise ValueError("data_range must be finite and positive")
    size = min(*x.shape, 11)
    size = size if size % 2 else size - 1
    axis = np.arange(size, dtype=np.float64) - size // 2
    gaussian = np.exp(-(axis ** 2) / (2 * 1.5 ** 2))
    gaussian /= gaussian.sum()
    kernel = gaussian[:, None] * gaussian[None, :]
    smooth = lambda image: correlate(image, kernel, mode="mirror")
    ux, uy = smooth(x), smooth(y)
    vx = np.maximum(smooth(x * x) - ux * ux, 0)
    vy = np.maximum(smooth(y * y) - uy * uy, 0)
    vxy = smooth(x * y) - ux * uy
    c1, c2 = (.01 * data_range) ** 2, (.03 * data_range) ** 2
    score = ((2 * ux * uy + c1) * (2 * vxy + c2)
             / ((ux * ux + uy * uy + c1) * (vx + vy + c2)))
    border = size // 2
    if border:
        score = score[border:-border, border:-border]
    return float(np.clip(score, -1, 1).mean())


def _image_pair(reference: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(reference, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if x.ndim != 2 or x.shape != y.shape or min(x.shape) < 1:
        raise ValueError(f"Expected matching grayscale images, got {x.shape} and {y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Image contains nonfinite pixels")
    if x.min() < 0 or x.max() > 1 or y.min() < 0 or y.max() > 1:
        raise ValueError("Images must use the fixed [0,1] range")
    return x, y


def threshold_crossing(cers: Mapping[str, float], severity_order: list[str],
                       threshold: float = .10) -> dict:
    """First *tested* severity with CER > threshold; no interpolation/extrapolation."""
    missing = [severity for severity in severity_order if severity not in cers]
    if missing:
        return {"status": "incomplete", "severity": None, "index": None,
                "missing_severities": missing}
    for index, severity in enumerate(severity_order):
        if cers[severity] > threshold:
            return {"status": "already_at_reference" if index == 0 else "crossed",
                    "severity": severity, "index": index}
    return {"status": "not_reached", "severity": None, "index": None}


def seed_maximum_summary(seed_cers: Mapping, shifted_severities: list[str]) -> dict:
    """Compute mean_j(max_s CER[j,s]), plus sample SD across training seeds."""
    if not shifted_severities or not seed_cers:
        raise ValueError("Need shifted severities and at least one seed")
    maxima = {}
    for seed, values in seed_cers.items():
        if any(severity not in values for severity in shifted_severities):
            raise ValueError(f"Seed {seed} has incomplete shifted conditions")
        maxima[seed] = max(float(values[severity]) for severity in shifted_severities)
    numbers = np.asarray(list(maxima.values()), dtype=float)
    return {"seed_maxima": maxima, "n_seeds": len(numbers),
            "mean_max_shifted_cer": float(numbers.mean()),
            "sd_max_shifted_cer": float(numbers.std(ddof=1)) if len(numbers) > 1 else None}
