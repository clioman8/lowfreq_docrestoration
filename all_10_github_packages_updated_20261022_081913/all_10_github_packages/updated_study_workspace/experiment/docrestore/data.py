"""Book-disjoint synthetic printed-text data, with exact rendering transcripts.

The official SLR11 corpus can be read directly from a tar archive without
extracting it. Text filtering uses fixed character/content rules and never OCR
results. The Latin-script ratio is NOT an English-language detector. Select an
explicitly verified English book allowlist for final experiments. Source
grouping precedes rendering or degradation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import tarfile
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .assets import REGULAR_FONTS, file_digest, safe_member_name


DEFAULTS = {
    "image_size": 256, "font_size": 22, "margin": 16,
    "samples_per_split": {"train": 1000, "validation": 100, "test": 100},
    "split_seed": 20261021, "render_seed": 20261022,
    "max_books": 500, "max_paragraphs_per_book": 300,
    "min_words": 8, "min_english_ratio": 0.85,
}
SPLITS = ("train", "validation", "test")
_TRANSLATE = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'",
                           "–": "-", "—": "-", "…": "...", "\u00a0": " "})


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _book_id(path: str) -> str:
    p = safe_member_name(path)
    return p.parent.name if p.parent.name == p.stem else p.stem


def iter_corpus_books(path: str | Path, max_book_bytes: int = 16 * 1024 * 1024):
    """Yield ``(book_id, text)`` from a corpus directory or tar archive.

    Directories are enumerated lexically. Archives retain their member order;
    this order and any ``max_books`` subset are recorded in the manifest.
    """
    path = Path(path)
    if path.is_dir():
        root = path.resolve()
        for text_file in sorted(path.rglob("*.txt")):
            if text_file.is_symlink() or root not in text_file.resolve().parents:
                raise ValueError(f"Corpus links are not allowed: {text_file}")
            if text_file.stat().st_size > max_book_bytes:
                raise ValueError(f"Text file exceeds max_book_bytes: {text_file}")
            yield _book_id(text_file.relative_to(path).as_posix()), _decode(text_file.read_bytes())
    else:
        with tarfile.open(path, "r|*") as archive:
            for member in archive:
                safe_member_name(member.name)
                if member.issym() or member.islnk():
                    raise ValueError(f"Corpus links are not allowed: {member.name}")
                if not member.isfile() or not member.name.lower().endswith(".txt"):
                    continue
                if member.size > max_book_bytes:
                    raise ValueError(f"Text file exceeds max_book_bytes: {member.name}")
                source = archive.extractfile(member)
                if source is not None:
                    yield _book_id(member.name), _decode(source.read())


def strip_gutenberg_boilerplate(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"^.*\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*$",
                      text, re.I | re.M)
    if start:
        text = text[start.end():]
    end = re.search(r"^(?:.*\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*|"
                    r"End of (?:the )?Project Gutenberg.*)$", text, re.I | re.M)
    if end:
        text = text[:end.start()]
    return text.strip()


def normalize_paragraph(text: str) -> str:
    """Use ASCII printables; preserve case and ordinary punctuation."""
    text = unicodedata.normalize("NFKD", text.translate(_TRANSLATE))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = "".join(c if (32 <= ord(c) <= 126 or c.isspace()) else " " for c in text)
    return " ".join(text.split())


def paragraphs_for_book(text: str, min_words: int = 8,
                        min_english_ratio: float = 0.85, limit: int = 300) -> list[str]:
    """Heuristic Latin/ASCII filter, not a claim of automatic language verification."""
    paragraphs = []
    seen = set()
    for raw in re.split(r"\n\s*\n", strip_gutenberg_boilerplate(text)):
        letters = [c for c in raw if c.isalpha()]
        latin = sum("LATIN" in unicodedata.name(c, "") for c in letters)
        if not letters or latin / len(letters) < min_english_ratio:
            continue
        normalized = normalize_paragraph(raw)
        words = re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", normalized)
        key = _sha(normalized)
        if len(words) < min_words or key in seen:
            continue
        seen.add(key)
        # Bounded chunks avoid giant paragraphs and provide deterministic candidates.
        tokens = normalized.split()
        for offset in range(0, len(tokens), 160):
            part = " ".join(tokens[offset:offset + 160])
            if len(re.findall(r"[A-Za-z]+", part)) >= min_words:
                paragraphs.append(part)
                if len(paragraphs) >= limit:
                    return paragraphs
    return paragraphs


def split_books(book_ids, seed: int = 20261021,
                ratios=(0.8, 0.1, 0.1), group_ids: dict[str, str] | None = None) -> dict[str, str]:
    """Assign source groups once, guaranteeing three nonempty splits when possible."""
    if len(ratios) != 3 or any(x < 0 for x in ratios) or sum(ratios) <= 0:
        raise ValueError("Expected three nonnegative split ratios")
    groups = defaultdict(list)
    for book in sorted(set(book_ids)):
        groups[(group_ids or {}).get(book, book)].append(book)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("At least three independent books/work groups are required")
    random.Random(seed).shuffle(keys)
    total = len(keys)
    fractions = [float(x) / sum(ratios) for x in ratios]
    n_val = max(1 if fractions[1] > 0 else 0, math.floor(total * fractions[1]))
    n_test = max(1 if fractions[2] > 0 else 0, math.floor(total * fractions[2]))
    n_train = total - n_val - n_test
    if n_train < 1:
        raise ValueError("Split ratios leave no training group")
    boundaries = (n_train, n_train + n_val)
    output = {}
    for index, group in enumerate(keys):
        split = SPLITS[0 if index < boundaries[0] else 1 if index < boundaries[1] else 2]
        for book in groups[group]:
            output[book] = split
    return output


def render_text_block(text: str, font_path: str | Path, image_size: int = 256,
                      font_size: int = 22, margin: int = 16):
    """Render whole words and complete lines; return image and exact drawn GT."""
    if image_size <= 2 * margin or margin < 0 or font_size < 1:
        raise ValueError("Invalid rendering geometry")
    font = ImageFont.truetype(str(font_path), font_size)
    image = Image.new("L", (image_size, image_size), 255)
    draw = ImageDraw.Draw(image)
    max_width = image_size - 2 * margin
    lines = []
    current = []
    for token in normalize_paragraph(text).split():
        token_box = draw.textbbox((0, 0), token, font=font, anchor="lt")
        if token_box[2] - token_box[0] > max_width:
            raise ValueError("A token is too wide for the fixed rendering geometry")
        candidate = " ".join(current + [token])
        candidate_box = draw.textbbox((0, 0), candidate, font=font, anchor="lt")
        if current and candidate_box[2] - candidate_box[0] > max_width:
            lines.append(" ".join(current))
            current = [token]
        else:
            current.append(token)
    if current:
        lines.append(" ".join(current))
    line_step = math.ceil(font_size * 1.35)
    drawn, boxes = [], []
    for index, line in enumerate(lines):
        y = margin + index * line_step
        bearing = draw.textbbox((0, 0), line, font=font, anchor="lt")[0]
        x = margin - bearing
        bbox = draw.textbbox((x, y), line, font=font, anchor="lt")
        if bbox[3] > image_size - margin:
            break
        if bbox[0] < margin or bbox[2] > image_size - margin:
            raise ValueError("Font bearings exceed the rendering bounds")
        draw.text((x, y), line, font=font, fill=0, anchor="lt")
        drawn.append(line)
        boxes.append(list(bbox))
    if not drawn:
        raise ValueError("No complete text line fits")
    bbox = [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]
    return image, "\n".join(drawn), drawn, bbox


def _jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                            for row in rows), encoding="utf-8")


def _work_mapping(config: dict) -> dict[str, str]:
    value = config.get("work_id_mapping", {})
    if isinstance(value, str):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("work_id_mapping must be a dictionary or JSON path")
    return {str(k): str(v) for k, v in value.items()}


def _book_allowlist(config: dict) -> set[str] | None:
    value = config.get("book_allowlist")
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("book_allowlist must be a nonempty list of IDs or a JSON-list path")
    if not all(isinstance(item, (str, int)) and not isinstance(item, bool) for item in value):
        raise ValueError("book_allowlist entries must be book IDs")
    return {str(item) for item in value}


def _reuse_source_split(path: str | Path, book_ids, book_hashes: dict,
                        group_ids: dict) -> tuple[dict[str, str], dict]:
    """Keep the pilot source partition fixed when increasing rendered samples."""
    path = Path(path)
    previous = json.loads(path.read_text(encoding="utf-8"))
    assignments = previous.get("book_assignments", {})
    expected_ids = set(book_ids)
    if set(assignments) != expected_ids:
        raise ValueError("source_split_manifest selected book IDs differ from the current corpus pool")
    if previous.get("book_normalized_sha256") != book_hashes:
        raise ValueError("source_split_manifest normalized book hashes differ from the current corpus")
    if any(value not in SPLITS for value in assignments.values()):
        raise ValueError("source_split_manifest contains an invalid split assignment")
    old_groups = previous.get("source_group_ids", {})
    if set(old_groups) != expected_ids:
        raise ValueError("source_split_manifest is missing source group membership")

    def partitions(groups):
        members = defaultdict(set)
        for book, group in groups.items():
            members[group].add(book)
        return {frozenset(value) for value in members.values()}

    if partitions(old_groups) != partitions(group_ids):
        raise ValueError("source_split_manifest has incompatible book/work group membership")
    group_splits = defaultdict(set)
    for book in expected_ids:
        group_splits[group_ids[book]].add(assignments[book])
    if any(len(values) != 1 for values in group_splits.values()):
        raise ValueError("source_split_manifest splits a book/work group across partitions")
    provenance = {
        "source_split_manifest": str(path.resolve()),
        "source_split_manifest_sha256": file_digest(path),
        "split_seed": previous.get("split_seed"),
        "split_method": "reused_source_manifest",
    }
    return assignments, provenance


def build_dataset(corpus: str | Path, fonts_dir: str | Path,
                   output: str | Path, config: dict) -> list[dict]:
    settings = {**DEFAULTS, **config.get("data", {})}
    allowlist = _book_allowlist(settings)
    language_verified = settings.get("language_validation", False)
    if not isinstance(language_verified, bool):
        raise ValueError("language_validation must be a boolean verification gate")
    if language_verified and allowlist is None:
        raise ValueError("language_validation=True requires an explicit English book_allowlist")
    validation_method = settings.get("language_validation_method")
    if validation_method is not None:
        if not isinstance(validation_method, str) or not validation_method.strip():
            raise ValueError("language_validation_method must be a nonempty string")
        validation_method = validation_method.strip()
    language_validation = ("metadata_verified_english_allowlist" if language_verified and validation_method
                           else "user_verified_english_allowlist" if language_verified
                           else "unverified_latin_script_heuristic_only")
    language_validation_is_user_assertion = language_verified and validation_method is None
    output, fonts_dir = Path(output), Path(fonts_dir)
    if (output / "manifest.jsonl").exists() and not settings.get("overwrite", False):
        raise FileExistsError(f"Dataset manifest already exists: {output / 'manifest.jsonl'}")
    names = settings.get("fonts", REGULAR_FONTS)
    fonts = [fonts_dir / name for name in names]
    if not fonts or any(not font.is_file() for font in fonts):
        raise FileNotFoundError("Prepare the requested fonts first with docrestore.assets fonts")
    font_hashes = {str(font): file_digest(font) for font in fonts}
    books: dict[str, list[str]] = {}
    book_hashes = {}
    source_order = []
    for book_id, text in iter_corpus_books(corpus):
        if allowlist is not None and book_id not in allowlist:
            continue
        parts = paragraphs_for_book(text, int(settings["min_words"]),
                                    float(settings["min_english_ratio"]),
                                    int(settings["max_paragraphs_per_book"]))
        if not parts:
            continue
        if book_id in books:
            raise ValueError(f"Duplicate source ID in corpus: {book_id}")
        books[book_id] = parts
        book_hashes[book_id] = _sha(normalize_paragraph(strip_gutenberg_boilerplate(text)))
        source_order.append(book_id)
        # An explicit allowlist selects all its IDs and takes precedence over the
        # convenience max_books limit used for an unverified development subset.
        if allowlist is not None and len(books) == len(allowlist):
            break
        if allowlist is None and settings.get("max_books") and len(books) >= int(settings["max_books"]):
            break
    if allowlist is not None and allowlist != books.keys():
        missing = sorted(allowlist - books.keys())
        raise ValueError("Allowed books missing or without eligible text: " + ", ".join(missing))
    # Work mappings and identical full text must both remain in a single group.
    explicit_work_ids = _work_mapping(settings)
    parents = {book: book for book in books}

    def find(book):
        while parents[book] != book:
            parents[book] = parents[parents[book]]
            book = parents[book]
        return book

    def union(left, right):
        a, b = sorted((find(left), find(right)))
        parents[b] = a

    work_owners, text_owners = {}, {}
    for book in sorted(books):
        work = explicit_work_ids.get(book, book)
        if work in work_owners:
            union(book, work_owners[work])
        work_owners[work] = book
        digest = book_hashes[book]
        if digest in text_owners:
            union(book, text_owners[digest])
        text_owners[digest] = book
    group_ids = {book: find(book) for book in books}
    if settings.get("source_split_manifest"):
        assignments, split_provenance = _reuse_source_split(
            settings["source_split_manifest"], books, book_hashes, group_ids)
    else:
        assignments = split_books(books, int(settings["split_seed"]), group_ids=group_ids)
        split_provenance = {"source_split_manifest": None,
                            "source_split_manifest_sha256": None,
                            "split_seed": settings["split_seed"],
                            "split_method": "seeded_group_split"}
    # Remove repeated normalized candidate text globally, before choosing samples.
    candidates = defaultdict(list)
    owners = set()
    duplicate_candidates_removed = 0
    for book in sorted(books):
        for part in books[book]:
            key = _sha(part)
            if key in owners:
                duplicate_candidates_removed += 1
                continue
            owners.add(key)
            candidates[assignments[book]].append((book, part))
    requested = settings["samples_per_split"]
    output.mkdir(parents=True, exist_ok=True)
    rows, drawn_owners, drawn_variants = [], {}, set()
    for split_index, split in enumerate(SPLITS):
        count = int(requested.get(split, 0))
        if count < 0:
            raise ValueError("samples_per_split cannot be negative")
        pool = list(candidates[split])
        if count and not pool:
            raise ValueError(f"No independent text candidates remain for {split}")
        rng = random.Random(int(settings["render_seed"]) + split_index)
        rng.shuffle(pool)
        accepted, attempts = 0, 0
        while accepted < count and attempts < max(100, 40 * count):
            book, part = pool[attempts % len(pool)]
            candidate_sha256 = _sha(part)
            source_word_offset = 0
            font = fonts[rng.randrange(len(fonts))]
            attempt_seed = rng.randrange(2 ** 32)
            attempts += 1
            # Rotate text at a deterministic whole-word boundary for subsequent passes.
            words = part.split()
            if attempts > len(pool) and len(words) > int(settings["min_words"]) + 10:
                start = random.Random(attempt_seed).randrange(
                    max(1, len(words) - int(settings["min_words"])))
                part = " ".join(words[start:])
                source_word_offset = start
            try:
                image, gt, lines, bbox = render_text_block(
                    part, font, int(settings["image_size"]),
                    int(settings["font_size"]), int(settings["margin"]))
            except ValueError:
                continue
            key = _sha(" ".join(gt.split()))
            if key in drawn_owners and drawn_owners[key] != split:
                continue
            variant_key = (key, font_hashes[str(font)])
            if variant_key in drawn_variants:
                continue
            drawn_owners[key] = split
            drawn_variants.add(variant_key)
            accepted += 1
            sample_id = f"{split}-{accepted:06d}"
            relative = Path("clean") / split / f"{sample_id}.png"
            (output / relative).parent.mkdir(parents=True, exist_ok=True)
            image.save(output / relative)
            rows.append({
                "sample_id": sample_id, "book_id": book,
                "work_id": explicit_work_ids.get(book, book),
                "source_group_id": group_ids[book], "split": split,
                "clean_path": relative.as_posix(), "gt_text": gt,
                "gt_normalized_sha256": key, "font_path": str(font.resolve()),
                "font_sha256": font_hashes[str(font)], "render_seed": attempt_seed,
                "image_width": image.width, "image_height": image.height,
                "geometry": {"width": image.width, "height": image.height,
                             "font_size": int(settings["font_size"]),
                             "margin": int(settings["margin"])},
                "lines": lines, "bbox": bbox,
                "clean_sha256": file_digest(output / relative),
                "candidate_sha256": candidate_sha256,
                "source_word_offset": source_word_offset,
                "language_validation": language_validation,
                "language_validation_method": validation_method,
            })
        if accepted != count:
            raise ValueError(f"Only {accepted}/{count} samples could be rendered for {split}; "
                             "increase the independent source pool or reduce the requested count")
    _jsonl(output / "manifest.jsonl", rows)
    split_manifest = {
        **split_provenance, "book_assignments": assignments,
        "source_group_ids": group_ids, "explicit_work_ids": explicit_work_ids,
        "book_normalized_sha256": book_hashes, "source_ingestion_order": source_order,
    }
    (output / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        "format_version": 1, "corpus": str(Path(corpus).resolve()),
        "settings": settings, "config_sha256": _sha(json.dumps(config, sort_keys=True)),
        "selected_books": len(books), "duplicate_candidates_removed": duplicate_candidates_removed,
        "counts": {split: sum(r["split"] == split for r in rows) for split in SPLITS},
        "text_policy": "ASCII printables after punctuation mapping and NFKD; preserve case; "
                       "Latin-script heuristic only; no OCR-dependent filtering",
        "source_policy": "Book/work groups split before rendering; exact normalized candidate "
                         "duplicates removed; exact rendered transcripts cannot cross splits",
        "book_allowlist": sorted(allowlist) if allowlist is not None else None,
        "language_validation": language_validation,
        "language_validation_is_user_assertion": language_validation_is_user_assertion,
        "language_validation_method": validation_method,
        **split_provenance,
    }
    (output / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows


def prepare_eval_dataset(manifest: list[dict], output: str | Path, config: dict) -> list[dict]:
    """Generate only validation/test severities using the shared numerical operator."""
    from .operators import forward_numpy
    output = Path(output)
    degradation = config["degradation"]
    config_sha256 = _sha(json.dumps(config, sort_keys=True))
    severities = degradation["severities"]
    ids = [str(item["id"]) for item in severities]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in ids):
        raise ValueError("Severity IDs must be unique safe filename components")
    rows = []
    for row in manifest:
        if row["split"] not in ("validation", "test"):
            continue
        clean = np.asarray(Image.open(output / row["clean_path"]).convert("L"), dtype=np.float64) / 255.0
        for severity in severities:
            eta = {"kappa": float(severity["kappa"]), "alpha": float(severity["alpha"]),
                   "T": float(severity["T"]), "n_steps": int(degradation.get("n_steps", 8)),
                   "h": float(degradation.get("h", 1.0))}
            observed = np.asarray(forward_numpy(clean, **eta), dtype=np.float32)
            if observed.shape != clean.shape or not np.all(np.isfinite(observed)):
                raise ValueError("Forward operator produced an invalid evaluation image")
            path = Path("degraded") / row["split"] / str(severity["id"]) / f"{row['sample_id']}.png"
            (output / path).parent.mkdir(parents=True, exist_ok=True)
            array_path = path.with_suffix(".npy")
            np.save(output / array_path, observed, allow_pickle=False)
            Image.fromarray(np.rint(np.clip(observed, 0, 1) * 255).astype(np.uint8)).save(output / path)
            rows.append({**row, "severity_id": str(severity["id"]), "eta": eta,
                         **eta, "degraded_path": path.as_posix(),
                         "degraded_array_path": array_path.as_posix(),
                         "degraded_array_sha256": file_digest(output / array_path),
                         "degraded_sha256": file_digest(output / path),
                         "generation_config_sha256": config_sha256,
                         "observation_storage": "float32 NPY for inference; uint8 PNG for OCR/reference only",
                         "png_quantization": "round(clip(float32_array,0,1)*255) to uint8"})
    _jsonl(output / "eval_manifest.jsonl", rows)
    evaluation_metadata = {
        "config_sha256": config_sha256,
        "operator": "diffusion_reaction_backward_euler_zero_flux",
        "n_steps": int(degradation.get("n_steps", 8)),
        "h": float(degradation.get("h", 1.0)),
        "severities": severities,
        "counts": {split: sum(row["split"] == split for row in rows)
                   for split in ("validation", "test")},
        "inference_input": "degraded_array_path float32 NPY, without uint8 quantization",
        "ocr_reference_input": "degraded_path PNG, round(clip(array,0,1)*255) uint8",
        "severity_validation": "Numerical generation is not validation of informative OCR severity",
    }
    (output / "eval_metadata.json").write_text(
        json.dumps(evaluation_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dataset_metadata = output / "dataset_metadata.json"
    if dataset_metadata.exists():
        metadata = json.loads(dataset_metadata.read_text(encoding="utf-8"))
        metadata["evaluation"] = evaluation_metadata
        dataset_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--fonts-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prepare-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.overwrite:
        config.setdefault("data", {})["overwrite"] = True
    manifest = build_dataset(args.corpus, args.fonts_dir, args.output, config)
    result = {"clean_samples": len(manifest), "manifest": str(args.output / "manifest.jsonl")}
    if args.prepare_eval:
        result["evaluation_samples"] = len(prepare_eval_dataset(manifest, args.output, config))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
