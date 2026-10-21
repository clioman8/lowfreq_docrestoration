"""Aggregate OCR records without pooling severity conditions or training seeds."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

from .evaluate import finite_json, read_jsonl, sha256_file
from .metrics import corpus_cer, seed_maximum_summary, threshold_crossing


def _sample_sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _optional_mean(records, key):
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else None


def _reference_kind(record, method):
    mode = str(record.get("mode", "")).lower()
    label = str(method).lower()
    if mode == "clean" or label == "clean":
        return "clean"
    if mode == "degraded" or label == "degraded":
        return "degraded"
    return None


def aggregate_records(records, severity_order, shifted_severities=None, threshold=.10):
    """Return per-severity, per-seed, method, and paired-method summaries.

    Primary outcome: mean over training seeds of each seed's maximum *corpus*
    CER across the prespecified shifted conditions. A single reference run is
    never relabeled as a training-seed replication.
    """
    if not severity_order or len(set(severity_order)) != len(severity_order):
        raise ValueError("Need a nonempty, unique severity order")
    shifted = list(shifted_severities or severity_order[1:])
    if not shifted or any(severity not in severity_order for severity in shifted):
        raise ValueError("Shifted severities must be a nonempty subset of severity_order")
    records = list(records)
    if not records:
        raise ValueError("No evaluation records supplied")
    grouped = defaultdict(list)
    seen = set()
    provenance = set()
    method_kinds = {}
    model_provenances = defaultdict(set)
    for record in records:
        split = record.get("split", "test")
        method = str(record.get("method", record.get("mode", "unknown")))
        kind = _reference_kind(record, method)
        method_key = (split, method)
        if method_key in method_kinds and method_kinds[method_key] != kind:
            raise ValueError(f"Mixed reference/learned records for {method_key}")
        method_kinds[method_key] = kind
        seed = None if kind else record.get("seed")
        severity = str(record.get("severity_id", "clean"))
        identity = (split, method, seed, severity, str(record["sample_id"]))
        if identity in seen:
            raise ValueError(f"Duplicate evaluation record {identity}")
        seen.add(identity)
        grouped[(split, method, seed, severity)].append(record)
        if record.get("ocr_provenance"):
            provenance.add(json.dumps(record["ocr_provenance"], sort_keys=True))
        if record.get("restoration_provenance"):
            model_provenances[(split, method, seed)].add(
                json.dumps(record["restoration_provenance"], sort_keys=True))
    if len(provenance) > 1:
        raise ValueError("Mixed OCR provenance: aggregate only one frozen evaluation pipeline")
    for key, values in model_provenances.items():
        checkpoint_hashes = {json.loads(value).get("checkpoint_sha256") for value in values}
        checkpoint_hashes.discard(None)
        if len(checkpoint_hashes) > 1:
            raise ValueError(f"Mixed restoration checkpoints within run {key}")

    severity_rows = []
    curves = defaultdict(dict)
    sample_sets = defaultdict(dict)
    for (split, method, seed, severity), items in grouped.items():
        row = {"split": split, "method": method, "seed": seed,
               "severity_id": severity, "n_samples": len(items),
               "n_books": len({item.get("book_id", item.get("source_book_id"))
                               for item in items
                               if item.get("book_id", item.get("source_book_id")) is not None}),
               "gt_chars": sum(int(item["gt_chars"]) for item in items),
               "edit_distance": sum(int(item["edit_distance"]) for item in items),
               "corpus_cer": corpus_cer(items),
               "mean_psnr_db": _optional_mean(items, "psnr_db"),
               "mean_ssim": _optional_mean(items, "ssim")}
        severity_rows.append(row)
        curves[(split, method, seed)][severity] = row["corpus_cer"]
        sample_sets[(split, method, seed)][severity] = {
            str(item["sample_id"]): (int(item["gt_chars"]), item.get("gt_normalized"))
            for item in items}

    # Same held-out documents and targets must be used for all conditions/groups.
    split_targets = {}
    for key, by_severity in sample_sets.items():
        split = key[0]
        for targets in by_severity.values():
            if split not in split_targets:
                split_targets[split] = targets
            elif split_targets[split] != targets:
                raise ValueError(f"Unmatched evaluation sample/GT sets in split {split}")

    seed_rows = []
    method_curves = defaultdict(dict)
    for (split, method, seed), cers in curves.items():
        kind = method_kinds[(split, method)]
        complete = kind != "clean" and all(severity in cers for severity in shifted)
        crossing = ({"status": "not_applicable_clean_reference", "severity": None, "index": None}
                    if kind == "clean" else threshold_crossing(cers, list(severity_order), threshold))
        local_changes = {
            f"{before}_to_{after}": cers[after] - cers[before]
            for before, after in zip(severity_order[:-1], severity_order[1:])
            if kind != "clean" and before in cers and after in cers}
        row = {"split": split, "method": method, "seed": seed,
               "complete_shifted_conditions": complete,
               "max_shifted_cer": max(cers[s] for s in shifted) if complete else None,
               "mean_shifted_cer": float(np.mean([cers[s] for s in shifted])) if complete else None,
               "in_range_cer": cers.get(severity_order[0]) if kind != "clean" else None,
               "clean_reference_cer": cers.get("clean", cers.get(severity_order[0])) if kind == "clean" else None,
               "threshold": threshold, "crossing_status": crossing["status"],
               "crossing_severity": crossing["severity"],
               "crossing_index": crossing["index"], "local_cer_changes": local_changes}
        seed_rows.append(row)
        method_curves[(split, method)][seed] = cers

    method_rows = []
    for (split, method), seed_cers in method_curves.items():
        # A reference row with seed=None remains a single nonlearned reference.
        kind = method_kinds[(split, method)]
        reference = kind is not None or list(seed_cers) == [None]
        complete = {seed: values for seed, values in seed_cers.items()
                    if kind != "clean" and all(severity in values for severity in shifted)}
        all_complete = len(complete) == len(seed_cers)
        maxima = seed_maximum_summary(complete, shifted) if complete else None
        condition_summaries = {}
        for severity in severity_order:
            values = [cers[severity] for cers in seed_cers.values() if severity in cers]
            if values:
                condition_summaries[severity] = {
                    "mean_cer": float(np.mean(values)),
                    "seed_sd": None if reference else _sample_sd(values),
                    "n_runs": len(values)}
        method_rows.append({"split": split, "method": method,
                            "run_kind": "nonlearned_reference" if reference else "training_seeds",
                            "n_runs": len(seed_cers), "n_complete_runs": len(complete),
                            "n_training_seeds": 0 if reference else len(seed_cers),
                            "all_shifted_conditions_complete": all_complete,
                            "mean_max_shifted_cer": maxima["mean_max_shifted_cer"] if maxima and all_complete and not reference else None,
                            "sd_max_shifted_cer": maxima["sd_max_shifted_cer"] if maxima and all_complete and not reference else None,
                            "reference_max_shifted_cer": maxima["mean_max_shifted_cer"]
                            if maxima and all_complete and kind == "degraded" else None,
                            "clean_reference_cer": next(iter(seed_cers.values())).get("clean",
                                next(iter(seed_cers.values())).get(severity_order[0])) if kind == "clean" else None,
                            "severity_summary": condition_summaries})

    paired_rows, paired_summary = [], []
    splits = sorted({key[0] for key in method_curves})
    for split in splits:
        learned_methods = sorted(method for candidate_split, method in method_curves
                                 if candidate_split == split
                                 and method_kinds[(split, method)] is None
                                 and any(seed is not None for seed in method_curves[(split, method)]))
        for base, compared in itertools.combinations(learned_methods, 2):
            first, second = method_curves[(split, base)], method_curves[(split, compared)]
            differences = []
            for seed in sorted(set(first) & set(second), key=str):
                if seed is None or any(s not in first[seed] or s not in second[seed] for s in shifted):
                    continue
                delta = max(second[seed][s] for s in shifted) - max(first[seed][s] for s in shifted)
                differences.append(delta)
                paired_rows.append({"split": split, "base_method": base,
                                    "compared_method": compared, "seed": seed,
                                    "delta_max_shifted_cer": delta,
                                    "sign_convention": "negative_favors_compared_method"})
            if differences:
                paired_summary.append({"split": split, "base_method": base,
                                       "compared_method": compared, "n_paired_seeds": len(differences),
                                       "mean_delta_max_shifted_cer": float(np.mean(differences)),
                                       "sd_delta_max_shifted_cer": _sample_sd(differences)})
    order = {severity: index for index, severity in enumerate(severity_order)}
    severity_rows.sort(key=lambda item: (item["split"], item["method"], str(item["seed"]),
                                          order.get(item["severity_id"], -1)))
    seed_rows.sort(key=lambda item: (item["split"], item["method"], str(item["seed"])))
    method_rows.sort(key=lambda item: (item["split"], item["method"]))
    return {"primary_definition": "mean_over_seeds_of_max_over_shifted_conditions_of_corpus_CER",
            "severity_order": list(severity_order), "shifted_severities": shifted,
            "threshold": threshold, "normalization_note": "Read ocr_provenance; no extra normalization here",
            "ocr_provenance": json.loads(next(iter(provenance))) if provenance else None,
            "restoration_provenance": [
                {"split": split, "method": method, "seed": seed,
                 "records": [json.loads(value) for value in sorted(values)]}
                for (split, method, seed), values in sorted(model_provenances.items(), key=lambda item: str(item[0]))],
            "severity_metrics": severity_rows, "seed_summary": seed_rows,
            "method_summary": method_rows, "paired_differences": paired_rows,
            "paired_summary": paired_summary}


def write_csv(path, records):
    records = list(records)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        if not records:
            return
        fields = list(dict.fromkeys(key for record in records for key in record))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(finite_json(value), ensure_ascii=False,
                                             allow_nan=False) if isinstance(value, (dict, list))
                             else finite_json(value) for key, value in record.items()})


def plot_curves(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = summary["severity_order"]
    for split in sorted({row["split"] for row in summary["method_summary"]}):
        figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
        for row in summary["method_summary"]:
            if row["split"] != split:
                continue
            conditions = row["severity_summary"]
            if not any(severity in conditions for severity in order):
                continue
            means = [conditions.get(s, {}).get("mean_cer", math.nan) for s in order]
            sds = [conditions.get(s, {}).get("seed_sd") or 0 for s in order]
            axis.errorbar(range(len(order)), means, yerr=sds,
                          label=row["method"], marker="o", capsize=3)
        axis.set(xticks=list(range(len(order))), xticklabels=order,
                 xlabel="Predefined severity condition", ylabel="Corpus CER",
                 title=f"{split}: mean and sample SD across training seeds")
        axis.legend()
        axis.grid(alpha=.25)
        safe_split = "".join(char if char.isalnum() or char in "-_" else "_" for char in str(split))
        figure.savefig(Path(output_dir) / f"cer_by_severity_{safe_split}.png", dpi=180)
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    severities = config.get("degradation", {}).get("severities", config.get("severities"))
    if severities is None:
        raise ValueError("Config needs degradation.severities")
    severity_order = [item["id"] if isinstance(item, dict) else item for item in severities]
    analysis = config.get("analysis", {})
    summary = aggregate_records([record for path in args.results for record in read_jsonl(path)],
                                severity_order,
                                analysis.get("shifted_severities", severity_order[1:]),
                                float(analysis.get("threshold", .10)))
    summary["aggregation_provenance"] = {
        "config_path": str(Path(args.config).resolve()),
        "config_file_sha256": sha256_file(Path(args.config)), "config": config,
        "result_files": [{"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))}
                         for path in args.results]}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(finite_json(summary), ensure_ascii=False,
                                                   indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for key in ("severity_metrics", "seed_summary", "method_summary", "paired_differences", "paired_summary"):
        write_csv(output / f"{key}.csv", summary[key])
    if args.plot:
        plot_curves(summary, output)
    print(json.dumps({"output_dir": str(output), "n_records": sum(row["n_samples"] for row in summary["severity_metrics"]),
                      "method_summary": finite_json(summary["method_summary"])}, ensure_ascii=False,
                     allow_nan=False))


if __name__ == "__main__":
    main()
