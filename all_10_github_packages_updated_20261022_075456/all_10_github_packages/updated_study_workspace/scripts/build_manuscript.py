"""Build the completed English thesis solely from finished, verified experiments.

Run with the bundled PDF Python after the PDF authoring operation marker.
`--check-inputs` performs the complete evidence gate without authoring a PDF.
This script never trains a model or invents an unobserved result.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "experiment"
DEFAULT_OUTPUT = ROOT / "output/pdf/low_frequency_subspace_document_restoration.pdf"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def configuration_hash(config):
    """Use the same canonical-object hash as docrestore.common.config_hash."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path_from_record(value, base=PROJECT):
    result = Path(value)
    return result if result.is_absolute() else base / result


def validate_run_metadata(metadata, job, config_sha256, manifest_sha256, eval_manifest_sha256):
    require(metadata.get("stage") == "final", "Run metadata is not final")
    for key in ("group", "seed", "lambda"):
        require(metadata.get(key) == job[key], f"Run metadata identity differs for {key}")
    require(metadata.get("config_sha256") == config_sha256 and
            configuration_hash(metadata["config"]) == config_sha256,
            "Training run used a different frozen configuration")
    require(metadata.get("manifest_sha256") == manifest_sha256,
            "Training run used a different clean-image manifest")
    require(metadata.get("eval_manifest_sha256") == eval_manifest_sha256,
            "Training run used a different evaluation manifest")


def validate_restoration_record(record, expected):
    """Bind the evaluated prediction to the selected run, not just its label."""
    require(record.get("method") == expected["job"]["group"] and
            record.get("seed") == expected["job"]["seed"], "OCR method/seed differs from selected run")
    provenance = record.get("restoration_provenance")
    require(isinstance(provenance, dict), "Restored OCR record lacks checkpoint provenance")
    require(provenance.get("checkpoint_sha256") == expected["checkpoint_sha256"],
            "Restored OCR used a different checkpoint from the selected best.pt")
    require(provenance.get("training_metadata") == expected["metadata"],
            "Restored OCR training metadata differs from the selected run")
    require(provenance.get("selection_step") == expected["summary"]["best_step"],
            "Restored OCR checkpoint selection step differs from the training summary")
    require(provenance.get("selection_validation_score") == [
                expected["summary"]["best_validation_max_shifted_cer"],
                expected["summary"]["best_validation_mean_shifted_cer"]],
            "Restored OCR checkpoint validation score differs from the training summary")
    require(provenance.get("provenance_file_sha256") == expected["provenance_sha256"],
            "Restored OCR checkpoint provenance file changed")


def completed_study():
    """Fail before authoring unless all reported evidence is complete and frozen."""
    config_path = PROJECT / "configs/frozen.json"
    config = read(config_path)
    require(config.get("configuration_frozen") is True and
            config.get("calibration_status") == "VALIDATED_AND_FROZEN", "Study is not frozen")
    state = read(PROJECT / "study_status.json")
    require(state.get("status") in {"experiments_complete", "manuscript_complete", "complete"},
            "All experiments must finish before final manuscript generation")
    lock = read(PROJECT / "provenance/study_lock.json")
    require(digest(config_path) == lock["config_sha256"], "Frozen configuration hash changed")
    for relative, expected in lock["source_sha256"].items():
        require(digest(PROJECT / relative) == expected, f"Frozen source changed: {relative}")
    require(lock["test_images_rendered_before_freeze"] is False and
            lock["test_ocr_inspected_before_freeze"] is False, "Test isolation was not recorded")
    plan = read(PROJECT / "provenance/run_plan_selected.json")
    require(plan["phase2_status"] == "resolved", "Lambda selection is incomplete")
    jobs = plan["phase1_candidates"] + plan["phase2_remaining_seeds"]
    require(len(jobs) == plan["planned_executed_runs_with_seed0_reuse"], "Incomplete execution plan")
    selected = [job for job in plan["phase1_candidates"]
                if job["lambda"] == plan["selected_lambda"][job["group"]]]
    selected += plan["phase2_remaining_seeds"]
    seeds = config["training"]["seeds"]
    require(len(selected) == 3 * len(seeds) == plan["final_reported_runs"],
            "Selected repetitions are incomplete")
    clean_manifest = PROJECT / "data/main/manifest.jsonl"
    eval_manifest = PROJECT / "data/main/eval_manifest.jsonl"
    clean_manifest_sha256, eval_manifest_sha256 = digest(clean_manifest), digest(eval_manifest)
    frozen_config_hash = configuration_hash(config)
    selected_keys = {(job["group"], job["seed"]) for job in selected}
    require(selected_keys == {(group, seed) for group in "ABC" for seed in seeds},
            "Selected plan does not contain exactly the expected method/seed identities")
    summaries = []
    selected_evidence = {}
    for job in jobs:
        run = path_from_record(job["run_dir"])
        summary, metadata = read(run / "summary.json"), read(run / "run_metadata.json")
        require(summary["steps"] == config["training"]["max_steps"], f"Incomplete training: {run}")
        require(summary["stage"] == "final", f"Nonfinal run in final results: {run}")
        for key in ("group", "seed", "lambda"):
            require(summary[key] == job[key] == metadata[key], f"Run identity mismatch: {run}")
        validate_run_metadata(metadata, job, frozen_config_hash,
                              clean_manifest_sha256, eval_manifest_sha256)
        require((run / "best.pt").is_file(), f"Missing selected checkpoint: {run}")
        summaries.append({**summary, "run_dir": str(run),
                          "reported": any(job == candidate for candidate in selected)})
        if any(job == candidate for candidate in selected):
            prediction_dir = PROJECT / "results/main/predictions" / run.name
            selected_evidence[(job["group"], job["seed"])] = {
                "job": job, "summary": summary, "metadata": metadata,
                "checkpoint_sha256": digest(run / "best.pt"),
                "provenance_sha256": digest(prediction_dir / "checkpoint_provenance.json"),
                "ocr_path": (prediction_dir / "ocr.jsonl").resolve()}
    aggregate_path = PROJECT / "results/main/aggregate/summary.json"
    aggregate = read(aggregate_path)
    for result in aggregate["aggregation_provenance"]["result_files"]:
        require(digest(path_from_record(result["path"])) == result["sha256"],
                "An aggregated OCR result changed")
    require(aggregate["aggregation_provenance"]["config_file_sha256"] == digest(config_path),
            "Aggregation used a different frozen configuration")
    clean_rows = jsonl(clean_manifest)
    test_rows = [row for row in clean_rows if row["split"] == "test"]
    test_targets = {row["sample_id"]: row["gt_text"] for row in test_rows}
    require(len(test_rows) == len(test_targets) == config["data"]["samples_per_split"]["test"],
            "Main test target manifest is incomplete or duplicated")
    references = PROJECT / "results/main/references"
    expected_results = {(references / "clean_ocr.jsonl").resolve(): ("clean", None),
                        (references / "degraded_ocr.jsonl").resolve(): ("degraded", None)}
    expected_results.update({value["ocr_path"]: key for key, value in selected_evidence.items()})
    actual_results = [path_from_record(result["path"]).resolve()
                      for result in aggregate["aggregation_provenance"]["result_files"]]
    require(len(actual_results) == len(expected_results) and set(actual_results) == set(expected_results),
            "Aggregated result paths differ from the selected runs and test references")
    severity_ids = [severity["id"] for severity in config["degradation"]["severities"]]
    for result_path in actual_results:
        method, seed = expected_results[result_path]
        expected_pairs = {(sample_id, severity) for sample_id in test_targets
                          for severity in (["clean"] if method == "clean" else severity_ids)}
        observed_pairs = set()
        with result_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                pair = (record["sample_id"], record.get("severity_id", "clean"))
                require(pair in expected_pairs and pair not in observed_pairs,
                        "Final OCR contains a missing, duplicate or non-test sample/condition")
                observed_pairs.add(pair)
                require(record.get("split") == "test" and record.get("method") == method and
                        record.get("seed") == seed, "Final OCR method/seed/split identity differs")
                require(record.get("gt_text") == test_targets[pair[0]],
                        "Final OCR ground truth differs from the main test manifest")
                require(record.get("ocr_provenance") == aggregate["ocr_provenance"],
                        "Final OCR pipeline differs from its aggregate provenance")
                if method in {"A", "B", "C"}:
                    require(record.get("mode") == "restored", "Learned result is not restored OCR")
                    validate_restoration_record(record, selected_evidence[(method, seed)])
                else:
                    require(record.get("mode") == method, "Reference OCR mode differs")
        require(observed_pairs == expected_pairs, "Final OCR is missing expected test samples or conditions")
    methods = {row["method"]: row for row in aggregate["method_summary"] if row["split"] == "test"}
    require({"A", "B", "C", "clean", "degraded"} <= set(methods), "Missing test methods/references")
    for method in "ABC":
        require(methods[method]["n_training_seeds"] == len(seeds) and
                methods[method]["all_shifted_conditions_complete"], f"Incomplete results for {method}")
    expected_n = config["data"]["samples_per_split"]["test"]
    for row in aggregate["severity_metrics"]:
        if row["split"] == "test":
            require(row["n_samples"] == expected_n, "Unequal/missing test documents")
    metadata = read(PROJECT / "data/main/dataset_metadata.json")
    require(metadata["counts"] == config["data"]["samples_per_split"], "Dataset counts disagree")
    split = read(PROJECT / "data/main/split_manifest.json")
    require(split["source_split_manifest_sha256"] == lock["source_split_sha256"],
            "Source split differs from the pretest frozen split")
    previous_split = read(path_from_record(split["source_split_manifest"]))
    require(digest(path_from_record(split["source_split_manifest"])) == lock["source_split_sha256"],
            "Pretest source split changed")
    for field in ("book_assignments", "source_group_ids", "book_normalized_sha256"):
        require(split[field] == previous_split[field], f"Source partition field changed: {field}")
    cases = read(PROJECT / "results/main/cases.json")
    require(cases["split"] == "test" and cases["example_seed"] == config["analysis"]["example_seed"],
            "Case analysis used a different split or seed")
    numerical = read(PROJECT / "provenance/numerical_verification.json")
    recovery_path = PROJECT / "provenance/recovery_events.json"
    recovery_events = read(recovery_path) if recovery_path.is_file() else []
    require(isinstance(recovery_events, list) and
            all(isinstance(event, dict) for event in recovery_events),
            "Recovery provenance must contain a list of recorded events")
    manuscript = read(ROOT / "analysis/manuscript_sections.json")
    curve = PROJECT / "results/main/aggregate/cer_by_severity_test.png"
    require(curve.is_file(), "Missing actual CER figure")
    return {"config": config, "lock": lock, "plan": plan, "summaries": summaries,
            "aggregate": aggregate, "methods": methods, "metadata": metadata,
            "split": split, "clean_rows": clean_rows, "cases": cases,
            "numerical": numerical, "manuscript": manuscript, "curve": curve,
            "aggregate_path": aggregate_path, "recovery_events": recovery_events,
            "recovery_path": recovery_path if recovery_path.is_file() else None}


def percent(value, decimals=3):
    return f"{100 * float(value):.{decimals}f}"


def number(value, decimals=3):
    if value in ("inf", "-inf"):
        return str(value)
    return f"{float(value):.{decimals}f}"


def mean_sd(values):
    values = list(values)
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.


def plain(value):
    return str(value).translate(str.maketrans({"\u2011": "-", "\u2012": "-", "\u2013": "-",
                                             "\u2014": "-", "\u2212": "-", "\u00a0": " "}))


def recovery_cost_paragraphs(events):
    """Describe only recorded recovery facts, without estimating missing time."""
    if not events:
        return []
    paragraphs = [
        "Reported timing totals sum the thirteen completed run summaries. They exclude "
        "the recorded abandoned attempts and downtime, and therefore are not an end-to-end "
        "elapsed study duration. No duration is estimated for the excluded work."
    ]
    for index, event in enumerate(events, start=1):
        details = []
        for label, keys in (("stage", ("stage",)), ("group", ("restart_group", "group")),
                            ("seed", ("restart_seed", "seed")),
                            ("lambda", ("restart_lambda", "lambda")),
                            ("last logged step", ("last_logged_step",))):
            for key in keys:
                if key in event:
                    details.append(f"{label} {event[key]}")
                    break
        sentence = f"Recovery record {index}: " + "; ".join(details) + "."
        if event.get("preserved_attempt"):
            sentence += " The abandoned attempt is preserved separately from the completed runs."
        if event.get("scientific_settings_changed") is False:
            sentence += " The record states that the scientific settings were unchanged."
        if event.get("reason"):
            sentence += " Recorded explanation: " + str(event["reason"])
        paragraphs.append(sentence)
    return paragraphs


def result_narrative(study):
    methods, aggregate = study["methods"], study["aggregate"]
    primary = {method: methods[method]["mean_max_shifted_cer"] for method in "ABC"}
    winner = min(primary, key=primary.get)
    contrasts = {(row["base_method"], row["compared_method"]): row
                 for row in aggregate["paired_summary"] if row["split"] == "test"}
    ca, cb = contrasts[("A", "C")], contrasts[("B", "C")]
    def direction(value):
        return "lower" if value < 0 else "higher" if value > 0 else "equal"
    sentence = (f"The mean maximum shifted-condition CER was "
                + ", ".join(f"{percent(primary[m])}% for {m}" for m in "ABC") + ". "
                + f"{winner} had the lowest observed value among the three objectives. "
                + f"For matched seeds, C minus A was {100*ca['mean_delta_max_shifted_cer']:+.3f} "
                + f"percentage points and C minus B was {100*cb['mean_delta_max_shifted_cer']:+.3f} "
                + "percentage points. These are descriptive differences from three paired seeds; "
                + "they do not establish statistical significance.")
    return primary, winner, ca, cb, sentence


def render_equations(study, temporary):
    jobs, images = [], {}
    def visit(node):
        for equation in node.get("equations", []):
            key = str(equation["id"])
            require(key not in images, f"Duplicate equation ID {key}")
            output = temporary / f"equation-{re.sub(r'[^A-Za-z0-9_-]', '_', key)}.png"
            jobs.append({"latex": equation["latex"], "path": str(output)})
            images[key] = output
        for child in node.get("subsections", []):
            visit(child)
    for section in study["manuscript"]["sections"]:
        visit(section)
    if jobs:
        request = temporary / "equations.json"
        request.write_text(json.dumps(jobs, indent=2, ensure_ascii=False), encoding="utf-8")
        subprocess.run([str(ROOT / ".venv/Scripts/python.exe"),
                        str(ROOT / "scripts/render_manuscript_math.py"), str(request)], check=True,
                       cwd=ROOT)
    return images


def build_pdf(study, output):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Flowable, Image, KeepTogether, PageBreak,
                                   Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)
    from PIL import Image as PILImage

    temporary = ROOT / "tmp/pdfs/manuscript"
    temporary.mkdir(parents=True, exist_ok=True)
    equations = render_equations(study, temporary)
    font_paths = {"Thesis": "times.ttf", "Thesis-Bold": "timesbd.ttf",
                  "Thesis-Italic": "timesi.ttf", "Thesis-BoldItalic": "timesbi.ttf"}
    for name, filename in font_paths.items():
        source = Path("C:/Windows/Fonts") / filename
        require(source.is_file(), f"Missing publication font: {source}")
        pdfmetrics.registerFont(TTFont(name, str(source)))
    pdfmetrics.registerFontFamily("Thesis", normal="Thesis", bold="Thesis-Bold",
                                  italic="Thesis-Italic", boldItalic="Thesis-BoldItalic")
    width = A4[0] - 50 * mm
    styles = getSampleStyleSheet()
    specifications = {
        "Body": dict(fontName="Thesis", fontSize=10.5, leading=14.5, alignment=TA_JUSTIFY,
                     spaceAfter=7, allowWidows=0, allowOrphans=0),
        "TitleThesis": dict(fontName="Thesis-Bold", fontSize=19, leading=23,
                            alignment=TA_CENTER, spaceAfter=12),
        "SubtitleThesis": dict(fontName="Thesis-Italic", fontSize=11, leading=15,
                               alignment=TA_CENTER, spaceAfter=16),
        "SectionThesis": dict(fontName="Thesis-Bold", fontSize=13.5, leading=17,
                              spaceBefore=16, spaceAfter=9, keepWithNext=True),
        "SubsectionThesis": dict(fontName="Thesis-Bold", fontSize=11.5, leading=15,
                                 spaceBefore=11, spaceAfter=6, keepWithNext=True),
        "CaptionThesis": dict(fontName="Thesis", fontSize=9, leading=12,
                              spaceBefore=5, spaceAfter=10),
        "TableCaptionThesis": dict(fontName="Thesis-Bold", fontSize=9.5, leading=12.5,
                                   spaceBefore=11, spaceAfter=5, keepWithNext=True),
        "CellThesis": dict(fontName="Thesis", fontSize=8.7, leading=11, spaceAfter=0),
        "CellHeaderThesis": dict(fontName="Thesis-Bold", fontSize=8.7, leading=11, spaceAfter=0),
        "SmallThesis": dict(fontName="Thesis", fontSize=9, leading=12, spaceAfter=5),
        "ReferenceThesis": dict(fontName="Thesis", fontSize=9, leading=12, spaceAfter=5,
                                leftIndent=21, firstLineIndent=-21),
        "AuthorThesis": dict(fontName="Thesis", fontSize=11, leading=15,
                             alignment=TA_CENTER, spaceAfter=8),
        "AffiliationThesis": dict(fontName="Thesis", fontSize=8.7, leading=11.7,
                                  alignment=TA_CENTER, spaceAfter=4),
        "OCRThesis": dict(fontName="Thesis", fontSize=9, leading=12,
                          leftIndent=7, rightIndent=5, spaceAfter=7),
    }
    for name, values in specifications.items():
        styles.add(ParagraphStyle(name, **values))
    story = []
    table_no = figure_no = 0

    def markup(text, citations=True):
        value = escape(plain(text))
        if citations:
            value = re.sub(r"\[(\d+)\]", lambda match:
                           f'<link href="#ref{match.group(1)}" color="#173b62">[{match.group(1)}]</link>', value)
        return value

    def paragraph(text, style="Body"):
        rendered = markup(text)
        if style == "OCRThesis":
            rendered = rendered.replace("\n", "<br/>")
        return Paragraph(rendered, styles[style])

    def p(text, style="Body"):
        story.append(paragraph(text, style))

    def heading(text, level=1):
        story.append(paragraph(text, "SectionThesis" if level == 1 else "SubsectionThesis"))

    def image(path, maximum_width=width, maximum_height=None, natural_dpi=None):
        path = Path(path)
        require(path.is_file(), f"Missing figure asset {path}")
        with PILImage.open(path) as source:
            pixel_width, pixel_height = source.size
        scale = min(maximum_width / pixel_width,
                    maximum_height / pixel_height if maximum_height else float("inf"),
                    72/natural_dpi if natural_dpi else float("inf"))
        return Image(str(path), width=pixel_width * scale, height=pixel_height * scale)

    def figure(path, caption, maximum_height=105*mm):
        nonlocal figure_no
        figure_no += 1
        story.append(KeepTogether([image(path, maximum_height=maximum_height),
                                   paragraph(f"Figure {figure_no}. {caption}", "CaptionThesis")]))

    def table(caption, headers, rows, proportions=None, note=None):
        nonlocal table_no
        table_no += 1
        p(f"Table {table_no}. {caption}", "TableCaptionThesis")
        proportions = proportions or [1] * len(headers)
        widths = [width * proportion / sum(proportions) for proportion in proportions]
        cells = [[paragraph(cell, "CellHeaderThesis") for cell in headers]]
        cells += [[paragraph(cell, "CellThesis") for cell in row] for row in rows]
        grid = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        grid.setStyle(TableStyle([
            ("LINEABOVE", (0,0), (-1,0), .8, colors.black),
            ("LINEBELOW", (0,0), (-1,0), .45, colors.black),
            ("LINEBELOW", (0,-1), (-1,-1), .8, colors.black),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(grid)
        if note:
            p(note, "SmallThesis")
        story.append(Spacer(1, 5))

    def equation(equation_record):
        equation_id = str(equation_record["id"])
        body = image(equations[equation_id], width - 36, 29*mm, natural_dpi=260)
        row = Table([[body, paragraph(f"({equation_id})", "SmallThesis")]],
                    colWidths=[width-31,31])
        row.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                                ("ALIGN",(0,0),(0,0),"CENTER"),
                                ("ALIGN",(1,0),(1,0),"RIGHT"),
                                ("TOPPADDING",(0,0),(-1,-1),5),
                                ("BOTTOMPADDING",(0,0),(-1,-1),7)]))
        story.append(row)
        if equation_record.get("description"):
            p(equation_record["description"])

    def authored(node, depth=1):
        label = f"{node['number']}. {node['title']}" if depth == 1 else f"{node['number']} {node['title']}"
        heading(label, min(depth,2))
        for text in node.get("paragraphs", []):
            p(text)
        for record in node.get("equations", []):
            equation(record)
        for subsection in node.get("subsections", []):
            authored(subsection, depth+1)

    config, plan, meta = study["config"], study["plan"], study["metadata"]
    aggregate, methods = study["aggregate"], study["methods"]
    primary, winner, ca, cb, findings = result_narrative(study)
    manuscript = study["manuscript"]
    p(manuscript["title"], "TitleThesis")
    if manuscript.get("subtitle"):
        p(manuscript["subtitle"], "SubtitleThesis")
    authors = manuscript["authors"]
    p("   ".join(f"{author['name']} ({author['affiliation_index']})" for author in authors), "AuthorThesis")
    for affiliation in manuscript["affiliations"]:
        p(f"{affiliation['index']} {affiliation['text']}", "AffiliationThesis")
    p(" | ".join(author["email"] for author in authors), "AffiliationThesis")
    story.append(Spacer(1, 10))
    heading("Abstract")
    abstract = ("We compare supervised reconstruction (A), full-residual forward consistency (B), "
                "and low-frequency subspace consistency (C) for document restoration under a controlled "
                "diffusion-reaction degradation path. The architecture, source split, training budget, "
                "OCR engine, validation criteria, and paired seeds are shared. A metadata-selected pool "
                f"of {meta['selected_books']:,} English source books is partitioned by recorded catalog contributor/work "
                f"groups before {meta['counts']['train']:,} training, {meta['counts']['validation']:,} validation, "
                f"and {meta['counts']['test']:,} test images are rendered. Each final test document is evaluated "
                "at five predefined conditions, including four conditions outside the training parameter range. "
                "Hyperparameters and checkpoints are selected on validation documents evaluated at this same "
                "known severity path. " + findings + " The experiment characterizes performance on held-out "
                "documents within one synthetic degradation family; it does not establish generalization to "
                "unseen severity values, acquisition processes, languages, or semantic recovery.")
    p(abstract)
    p("Keywords: document image restoration; optical character recognition; low-frequency subspace; "
      "forward consistency; severity shift; controlled empirical comparison.", "SmallThesis")
    story.append(PageBreak())

    sections = {int(section["number"]): section for section in manuscript["sections"]}
    for section_number in range(1,6):
        require(section_number in sections, f"Missing authored section {section_number}")
        authored(sections[section_number])
        if section_number == 3:
            assignment_counts = Counter(study["split"]["book_assignments"].values())
            sampled_books = defaultdict(set)
            sampled_groups = defaultdict(set)
            source_groups = study["split"]["source_group_ids"]
            for row in study["clean_rows"]:
                book = str(row.get("book_id", row.get("source_book_id")))
                sampled_books[row["split"]].add(book)
                sampled_groups[row["split"]].add(source_groups[book])
            assigned_groups = defaultdict(set)
            for book, split_name in study["split"]["book_assignments"].items():
                assigned_groups[split_name].add(source_groups[book])
            table("Source partitions and rendered clean samples.",
                  ["Split", "Assigned books", "Assigned groups", "Sampled books", "Clean images"],
                  [[split_name.capitalize(), str(assignment_counts[split_name]),
                    str(len(assigned_groups[split_name])), str(len(sampled_books[split_name])),
                    f"{meta['counts'][split_name]:,}"] for split_name in ("train","validation","test")],
                  [1.1,1.1,1.1,1.1,1.1],
                  "Groups are recorded catalog contributor/work components; sampled books count only books "
                  "that actually supply rendered examples. Split identities are fixed before rendering.")
            table("Predefined evaluation path of the diffusion-reaction operator.",
                  ["Condition", "Diffusion kappa", "Reaction alpha", "Terminal T", "Status"],
                  [[s["id"], str(s["kappa"]), str(s["alpha"]), str(s["T"]),
                    "In training range" if s["id"] == "S0" else "Outside training range"]
                   for s in config["degradation"]["severities"]], [1,1.2,1.2,1,2])
        if section_number == 5:
            table("Frozen training, selection, and OCR settings.", ["Item", "Setting"], [
                ["Restoration model", "Common U-Net; channels " + ", ".join(map(str,config["model"]["channels"]))],
                ["Training budget", f"{config['training']['max_steps']:,} updates per run; batch {config['training']['batch_size']}"],
                ["Training seeds", ", ".join(map(str,config["training"]["seeds"]))],
                ["Learning rate", str(config["training"]["learning_rate"])],
                ["Loss constants", f"beta={config['loss']['beta']}; m={config['loss']['subspace_dim']}"],
                ["B/C lambda candidates", ", ".join(map(str,config["loss"]["lambda_grid"]))],
                ["Selected lambda", "; ".join(f"{method}={plan['selected_lambda'][method]}" for method in "ABC")],
                ["Checkpoint updates", ", ".join(map(str,config["training"]["eval_steps"]))],
                ["Selection criterion", "Minimum validation maximum shifted corpus CER, then mean shifted CER; earliest checkpoint on exact ties; smaller lambda for candidate ties"],
                ["OCR", f"{aggregate['ocr_provenance']['tesseract_version'].splitlines()[0]}; eng; OEM 1; PSM {config['ocr']['psm']}; workers {config['ocr']['workers']}"],
                ["OCR preprocessing", "No external preprocessing; frozen Tesseract internal preprocessing"],
                ["Threshold", f"CER > {config['analysis']['threshold']}; adjacent interval {config['analysis']['threshold_adjacent_interval']}"],
            ], [1.2,3.3])

    heading("6. Results")
    heading("6.1 OCR outcomes by severity", 2)
    p("All reported OCR comparisons use the same held-out test documents, exact transcripts, fixed "
      "Tesseract pipeline, and severity order. CER is a ratio and is reported below as percent. "
      "A clean or degraded reference is one nonlearned evaluation, so no training-seed SD is assigned to it.")
    p(f"Clean-reference corpus CER was {percent(methods['clean']['clean_reference_cer'])}%. "
      f"The degraded-input maximum over shifted conditions was "
      f"{percent(methods['degraded']['reference_max_shifted_cer'])}%.")
    severity_order = aggregate["severity_order"]
    curve_rows = []
    for method in ("degraded","A","B","C"):
        row = ["Input" if method == "degraded" else method]
        for severity in severity_order:
            item = methods[method]["severity_summary"][severity]
            cell = percent(item["mean_cer"])
            if item["seed_sd"] is not None:
                cell += " ± " + percent(item["seed_sd"])
            row.append(cell)
        curve_rows.append(row)
    table("Test corpus CER (%) by severity; learned methods show mean ± sample SD across seeds.",
          ["Method"] + severity_order, curve_rows, [1.1]+[1.2]*len(severity_order))
    shifted_mean_rows = []
    for method in "ABC":
        values = [next(row["mean_shifted_cer"] for row in aggregate["seed_summary"]
                       if row["split"] == "test" and row["method"] == method and row["seed"] == seed)
                  for seed in config["training"]["seeds"]]
        average, deviation = mean_sd(values)
        shifted_mean_rows.append([method] + [percent(value) for value in values] +
                                 [percent(average) + " ± " + percent(deviation)])
    table("Secondary outcome: mean shifted corpus CER for each training seed (%).",
          ["Method"] + [f"Seed {seed}" for seed in config["training"]["seeds"]] + ["Mean ± SD"],
          shifted_mean_rows, [1]*(len(config["training"]["seeds"])+2),
          "Each seed value is the arithmetic mean of corpus CER over the predefined shifted conditions "
          "S1-S4, read from the existing aggregation. The final column is the mean and sample SD across "
          "training seeds. This secondary endpoint does not replace the maximum-CER primary endpoint.")
    input_shifted_mean = next(row["mean_shifted_cer"] for row in aggregate["seed_summary"]
                              if row["split"] == "test" and row["method"] == "degraded")
    p(f"The degraded-input mean shifted corpus CER was {percent(input_shifted_mean)}%, "
      "from a single nonlearned reference evaluation; no training-seed SD is assigned to it.")
    figure(study["curve"], "Test severity curves from the frozen OCR aggregation. Error bars show "
           "sample SD across matched training seeds for learned methods; the input reference is not replicated.")

    heading("6.2 Maximum shifted CER and paired seeds", 2)
    p(findings)
    table("Primary endpoint: maximum shifted corpus CER for each training seed (%).",
          ["Method"] + [f"Seed {seed}" for seed in config["training"]["seeds"]] + ["Mean ± SD"],
          [[method] + [percent(next(row["max_shifted_cer"] for row in aggregate["seed_summary"]
                                   if row["split"] == "test" and row["method"] == method and row["seed"] == seed))
                       for seed in config["training"]["seeds"]] +
           [percent(methods[method]["mean_max_shifted_cer"]) + " ± " + percent(methods[method]["sd_max_shifted_cer"])]
           for method in "ABC"], [1]*5,
          "Each seed maximum is computed before averaging across seeds; it is not the maximum of the mean severity curve.")
    paired_rows = []
    for base in ("A","B"):
        contrast = ca if base == "A" else cb
        row = [f"C - {base}"]
        row += [f"{100*next(item['delta_max_shifted_cer'] for item in aggregate['paired_differences'] if item['split']=='test' and item['base_method']==base and item['compared_method']=='C' and item['seed']==seed):+.3f}"
                for seed in config["training"]["seeds"]]
        row += [f"{100*contrast['mean_delta_max_shifted_cer']:+.3f} ± {100*contrast['sd_delta_max_shifted_cer']:.3f}"]
        paired_rows.append(row)
    table("Paired differences in the primary endpoint (percentage points).",
          ["Contrast"]+[f"Seed {seed}" for seed in config["training"]["seeds"]]+["Mean ± SD"],
          paired_rows, [1]*5, "A negative difference favors C. Three paired seeds do not support a strong significance claim.")

    heading("6.3 Pixel fidelity and recognition transitions", 2)
    p("PSNR and SSIM are secondary image-fidelity measures. For each seed and severity they are "
      "averaged over documents; the table then summarizes those averages across training seeds. "
      "No weighted pooling across severity conditions is used.")
    fidelity_rows = []
    for severity in severity_order:
        for method in ("degraded","A","B","C"):
            records = [row for row in aggregate["severity_metrics"] if row["split"] == "test"
                       and row["method"] == method and row["severity_id"] == severity]
            cells = [severity, "Input" if method == "degraded" else method]
            for key in ("mean_psnr_db", "mean_ssim"):
                values = [float(row[key]) for row in records]
                if any(math.isinf(value) for value in values):
                    cell = "inf"
                else:
                    average, deviation = mean_sd(values)
                    cell = number(average, 3 if key == "mean_psnr_db" else 4)
                    if method != "degraded":
                        cell += " ± " + number(deviation, 3 if key == "mean_psnr_db" else 4)
                cells.append(cell)
            fidelity_rows.append(cells)
    table("Image fidelity at each test severity (PSNR in dB; SSIM dimensionless).",
          ["Severity", "Method", "PSNR", "SSIM"], fidelity_rows, [1,1,1.8,1.8])
    crossings = [row for row in aggregate["seed_summary"]
                 if row["split"] == "test" and row["method"] in {"A","B","C","degraded"}]
    table("First tested condition with corpus CER above the prespecified threshold.",
          ["Method", "Seed", "Crossing condition", "Status"],
          [["Input" if row["method"] == "degraded" else row["method"],
            "Reference" if row["seed"] is None else str(row["seed"]),
            row["crossing_severity"] or "None in tested path", row["crossing_status"].replace("_"," ")]
           for row in crossings], [1,1,1.5,2.2],
          "Crossing refers only to the discrete tested path; no continuous transition point is interpolated.")

    heading("6.4 Selection and computational cost", 2)
    p("The search executes the baseline and three lambda values for each consistency method at seed 0, "
      "then trains the selected setting at seeds 1 and 2. The selected seed-0 candidate is reused "
      "as the first reported repetition. Seven completed candidate runs plus six completed additional runs "
      "produce thirteen completed training runs and nine reported method-seed repetitions. "
      "Cost accounting includes the four unselected candidates.")
    table("Validation-selected settings and checkpoints.",
          ["Method", "Lambda", "Seed", "Selected step", "Val. max CER (%)", "Val. mean CER (%)"],
          [[row["group"], str(row["lambda"]), str(row["seed"]), str(row["best_step"]),
            percent(row["best_validation_max_shifted_cer"]), percent(row["best_validation_mean_shifted_cer"])]
           for row in sorted(study["summaries"], key=lambda r:(r["group"],r["seed"],r["lambda"]))],
          [0.8,0.8,0.7,1.1,1.4,1.4], "All thirteen completed search/training runs are listed; test results use only the selected nine.")
    selected_summaries = [row for row in study["summaries"] if row["reported"]]
    cost_rows = []
    for label, records in (("All completed runs", study["summaries"]), ("Reported selected runs", selected_summaries)):
        cost_rows.append([label, str(len(records)), f"{sum(r['steps'] for r in records):,}",
                          number(sum(r["total_seconds"] for r in records)/3600, 3),
                          number(sum(r["validation_seconds"] for r in records)/3600, 3),
                          number(sum(r["training_and_diagnostics_seconds"] for r in records)/3600, 3)])
    table("Measured training-stage wall-clock cost, including validation OCR.",
          ["Scope", "Runs", "Updates", "Total (h)", "Validation (h)", "Train + diagnostic (h)"],
          cost_rows, [1.7,.6,.9,1,1.2,1.4],
          f"Hardware: {study['numerical']['gpu']}. CPU logical processors: {study['numerical']['logical_processors']}. "
          "These sums cover run summaries, not data preparation, final test OCR, or manuscript generation.")
    for text in recovery_cost_paragraphs(study["recovery_events"]):
        p(text)

    heading("6.5 Deterministic case analysis", 2)
    p("Cases are selected by the frozen rules using seed 0: the smallest sample ID at S0, an input "
      "CER nearest the threshold within the validation-fixed interval, the largest input CER at S4, "
      "and the largest positive document-level C minus A CER. The panels and transcriptions are "
      "descriptive evidence and do not relabel OCR output as semantic recovery.")
    labels = {"reference":"Reference condition", "threshold_adjacent":"Threshold-adjacent input",
              "severe_input":"Severe input", "largest_C_deterioration_vs_A":"Largest C deterioration relative to A"}
    for key, case in study["cases"]["cases"].items():
        heading(labels.get(key,key), 2)
        if case["status"] != "selected":
            p("No test example met this prespecified selection rule.")
            continue
        p(f"Sample {case['sample_id']}; {case['severity_id']}; seed {study['cases']['example_seed']}. "
          f"Input CER {percent(case['input_cer'])}%; C minus A document CER "
          f"{100*case['C_minus_A_cer']:+.3f} percentage points.", "SmallThesis")
        panel_paths = [("Clean", path_from_record(case["input"]["clean_path"], PROJECT/"data/main"))]
        panel_paths += [("Input", Path(case["input"]["evaluated_path"]))]
        panel_paths += [(method, Path(case[method]["evaluated_path"])) for method in "ABC"]
        panel_width = (width-12)/3
        panels = [[paragraph(label,"CellHeaderThesis"), image(path,panel_width,panel_width)]
                  for label,path in panel_paths]
        rows = []
        for start in (0,3):
            batch = panels[start:start+3]
            rows.append([entry[0] for entry in batch] + [""]*(3-len(batch)))
            rows.append([entry[1] for entry in batch] + [""]*(3-len(batch)))
        panel_table = Table(rows,colWidths=[width/3]*3)
        panel_table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                                        ("ALIGN",(0,0),(-1,-1),"CENTER"),
                                        ("LEFTPADDING",(0,0),(-1,-1),2),
                                        ("RIGHTPADDING",(0,0),(-1,-1),2)]))
        figure_no += 1
        story.append(KeepTogether([panel_table, paragraph(
            f"Figure {figure_no}. {labels.get(key,key)}: clean, degraded input, and restoration by A, B, and C. "
            "All panels use the same document crop and scale.", "CaptionThesis")]))
        p("Ground truth: " + case["input"]["gt_text"], "OCRThesis")
        for method,label in (("input","Input"),("A","A"),("B","B"),("C","C")):
            record = case[method]
            p(f"{label} OCR (CER {percent(record['cer'])}%): " + record["ocr_text"], "OCRThesis")

    heading("7. Discussion")
    p(findings)
    c_vs_a = "reduces" if ca["mean_delta_max_shifted_cer"] < 0 else "increases" if ca["mean_delta_max_shifted_cer"] > 0 else "leaves unchanged"
    c_vs_b = "reduces" if cb["mean_delta_max_shifted_cer"] < 0 else "increases" if cb["mean_delta_max_shifted_cer"] > 0 else "leaves unchanged"
    p(f"Within this frozen experiment, restricting consistency to the selected low-frequency space "
      f"{c_vs_a} the observed mean maximum shifted CER relative to reconstruction alone and {c_vs_b} "
      "it relative to full-residual consistency. This conclusion concerns the trained objectives together "
      "with their validation-selected weights and checkpoints. It does not isolate an architecture change, "
      "because the architecture is common, or a universal property of low-frequency regularization.")
    p("The forward model matches data generation by construction and its parameters are available to the "
      "training loss. Full consistency constrains every residual component, whereas C constrains a fixed "
      "32-dimensional projection. Both consistency energies are divided by the full pixel count; the "
      "projected loss is not rescaled by the number of retained modes. The same numerical lambda "
      "need not imply equal gradient strength. Equal candidate "
      "budgets and reported gradient diagnostics make the comparison explicit, but cannot make all "
      "optimization effects identical.")
    p("Severity-wise CER, seed maxima, pixel fidelity, and the selected failure cases answer different "
      "questions. Improved PSNR or SSIM is not sufficient evidence of improved recognition. Likewise, "
      "a favorable maximum over four tested shifts does not imply monotonic behavior at every document "
      "or severity. The discrete threshold table describes the sampled path rather than estimating a "
      "continuous recognition boundary.")
    p("Validation and test use the same known S0-S4 parameter values. Test documents and linked source "
      "groups are held out, but severity values are deliberately used during validation selection. "
      "The findings therefore concern performance on held-out documents along an out-of-training-range "
      "parameter path, rather than generalization to previously unseen severity values.")

    heading("8. Limitations")
    if 8 in sections:
        for text in sections[8].get("paragraphs", []):
            p(text)
        for subsection in sections[8].get("subsections", []):
            authored(subsection,2)
    p("Only three matched seeds are reported. In addition, seed 0 participates in lambda selection and "
      "is reused in the final three-seed summary, so the three repetitions are not independent of the "
      "hyperparameter search. No independent postselection seed cohort or confidence interval is claimed.")
    p("The same severity parameters are used on validation and test. Thus the test split protects "
      "document/source identities while the parameter path remains known to selection. All degradations "
      "are synthetic and drawn from the same diffusion-reaction family; the matched forward model and "
      "available parameters favor controlled interpretation over acquisition realism.")
    p("The corpus is limited to English public-domain text, ASCII-oriented transcript preparation, "
      "three fixed Liberation Regular fonts at one scale, small grayscale crops, and one fixed OCR engine. Exact duplicate "
      "and source-group controls do not establish the absence of every paraphrase, quotation, or textual "
      "overlap. Generalization to multilingual documents, real scans, layouts, handwriting, and other "
      "OCR systems remains unmeasured.")
    p("The real Fourier basis is truncated at dimension 32; this cutoff retains an incomplete final "
      "sine/cosine pair. The implemented space remains orthonormal, but this choice introduces an "
      "orientation/phase convention and is not a complete isotropic frequency shell. No basis-dimension "
      "or alternative-basis ablation is included. The basis also differs from the cosine eigenbasis "
      "used to solve the zero-flux degradation operator.")

    heading("9. Conclusion")
    p("This study completed a controlled comparison of reconstruction-only, full-residual, and "
      "low-frequency subspace consistency losses under a shared document-restoration protocol. " + findings)
    p(f"The observed lowest primary value was obtained by {winner}. The evidence supports a bounded "
      "empirical statement about the specified corpus, forward operator, parameter path, basis, "
      "validation search, and fixed OCR engine. It does not justify a universal improvement claim or "
      "a claim of semantic recovery. Broader conclusions require independent postselection seeds, "
      "basis and weight ablations, and evaluation on real acquisition degradations.")

    heading("References")
    reference_ids = set()
    for reference in manuscript["references"]:
        reference_id = str(reference["id"])
        require(reference_id not in reference_ids, "Duplicate reference ID")
        reference_ids.add(reference_id)
        authors_text = reference.get("authors", "")
        if isinstance(authors_text,list):
            authors_text = ", ".join(authors_text)
        venue = reference.get("venue", "")
        if reference.get("volume"):
            venue += ", " + str(reference["volume"])
            if reference.get("issue"):
                venue += "(" + str(reference["issue"]) + ")"
        if reference.get("pages"):
            venue += ", pp. " + str(reference["pages"])
        pieces = [authors_text, str(reference.get("year","")), reference["title"], venue]
        text = ". ".join(piece.rstrip(".") for piece in pieces if piece) + "."
        url = reference.get("url")
        rendered = f'<a name="ref{reference_id}"/>[{reference_id}] ' + markup(text,False)
        if url:
            rendered += f' <link href="{escape(url,quote=True)}" color="#173b62">{escape(url)}</link>'
        story.append(Paragraph(rendered, styles["ReferenceThesis"]))

    story.append(PageBreak())
    heading("Appendix A. Numerical implementation verification")
    p("These checks verify numerical implementation properties and finite gradients on test tensors. "
      "They are not restoration-performance experiments or proof of a generalization advantage.")
    for appendix in manuscript.get("appendices", []):
        if appendix.get("number") == "B":
            for text in appendix.get("paragraphs", []):
                p(text)
    numerical = study["numerical"]
    table("Numerical checks for the implemented forward operator and subspace projector.",
          ["Check", "Observed value"], [
              ["Dense implicit Euler maximum absolute error", f"{numerical['dense_implicit_euler_max_abs_error']:.3e}"],
              ["Main basis orthogonality maximum absolute error", f"{numerical['main_basis_orthogonality_max_abs_error']:.3e}"],
              ["Projector idempotence relative error", f"{numerical['main_projector_idempotence_relative_error']:.3e}"],
              ["CUDA float32 versus NumPy maximum absolute error", f"{numerical['cuda_float32_numpy_max_abs_error']:.3e}"],
              ["CUDA float32 gradient finite / nonzero", f"{numerical['cuda_float32_finite_gradient']} / {numerical['cuda_float32_nonzero_gradient']}"],
              ["Basis shape", " x ".join(map(str,numerical["main_basis_shape"]))],
          ], [3,1.5])

    heading("Appendix B. Reproduction and evidence provenance")
    p("The evidence gate checks the frozen scientific source and configuration hashes, all completed "
      "training summaries, selected seeds, full test conditions, and hashes of the OCR files used by "
      "the aggregation. All image-level OCR outputs retain engine/model, transcript normalization, "
      "checkpoint, and evaluation-manifest provenance. Final figures and tables are generated from "
      "those files without manual replacement of result cells.")
    for appendix in manuscript.get("appendices", []):
        if appendix.get("number") == "A":
            for text in appendix.get("paragraphs", []):
                p(text)
    provenance_files = [("Frozen configuration", PROJECT/"configs/frozen.json"),
                        ("Source split", PROJECT/"data/main/split_manifest.json"),
                        ("Clean-image manifest", PROJECT/"data/main/manifest.jsonl"),
                        ("Evaluation manifest", PROJECT/"data/main/eval_manifest.jsonl"),
                        ("Selected run plan", PROJECT/"provenance/run_plan_selected.json"),
                        ("Aggregate summary", study["aggregate_path"]),
                        ("Case selections", PROJECT/"results/main/cases.json"),
                        ("Numerical verification", PROJECT/"provenance/numerical_verification.json")]
    if study["recovery_path"] is not None:
        provenance_files.append(("Operating recovery events", study["recovery_path"]))
    table("SHA-256 evidence fingerprints.", ["Artifact", "SHA-256"],
          [[label, digest(path)[:32]+" "+digest(path)[32:]] for label,path in provenance_files], [1.2,3.5])
    p("Frozen at " + study["lock"]["frozen_at_kst"] + ". OCR language model SHA-256: " +
      aggregate["ocr_provenance"]["tessdata_sha256"] + ".", "SmallThesis")
    p("Execution order: prepare the metadata-filtered source pool and source split; freeze train/validation "
      "calibration and scientific code; generate final data; train the seven seed-0 candidates; select "
      "lambda on validation; train six additional matched runs; predict and evaluate the nine selected "
      "test runs plus nonlearned references; aggregate; select deterministic cases; build the manuscript.")
    p("Executable entry points are scripts/prepare_study.py, scripts/freeze_study.py, scripts/run_study.py, "
      "and scripts/build_manuscript.py. The experiment/provenance directory retains the frozen lock, "
      "environment requirements, selected run plan, and numerical-verification record. All file paths "
      "and hashes are part of the accompanying reproducibility package.", "SmallThesis")

    def decoration(canvas, document):
        canvas.saveState()
        if document.page > 1:
            canvas.setFont("Thesis",8)
            canvas.setFillColor(colors.HexColor("#444444"))
            canvas.drawString(25*mm,A4[1]-15.5*mm,"Low-Frequency Subspace Consistency for Document Restoration")
            canvas.setStrokeColor(colors.HexColor("#bbbbbb"))
            canvas.setLineWidth(.35)
            canvas.line(25*mm,A4[1]-18*mm,A4[0]-25*mm,A4[1]-18*mm)
        canvas.setFont("Thesis",8.5)
        canvas.setFillColor(colors.black)
        canvas.drawCentredString(A4[0]/2,13*mm,str(document.page))
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_pdf = output.with_name(output.stem+".building.pdf")
    document = SimpleDocTemplate(str(temporary_pdf), pagesize=A4,
                                leftMargin=25*mm,rightMargin=25*mm,
                                topMargin=24*mm,bottomMargin=23*mm,
                                title=manuscript["title"],
                                author=", ".join(author["name"] for author in authors),
                                subject="Completed controlled document-restoration experiment",
                                pageCompression=1)
    try:
        document.build(story,onFirstPage=decoration,onLaterPages=decoration)
        from pypdf import PdfReader
        reader = PdfReader(temporary_pdf)
        require(len(reader.pages) >= 10, "Unexpectedly short thesis output")
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        for phrase in ("PRE-EXPERIMENTAL ROUGH DRAFT", "Manuscript status:", "TODO", "TBD"):
            require(phrase not in full_text, f"Unfinished manuscript marker: {phrase}")
        for identifier in re.findall(r"\[(\d+)\]", full_text):
            require(identifier in reference_ids, f"Undefined reference [{identifier}]")
        temporary_pdf.replace(output)
        report = {"output":str(output), "pages":len(reader.pages), "sha256":digest(output),
                  "tables":table_no,"figures":figure_no,
                  "source_aggregate_sha256":digest(study["aggregate_path"]),
                  "requires_visual_review":True}
        (output.parent / "manuscript_build.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(report))
    finally:
        if temporary_pdf.exists():
            temporary_pdf.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    study = completed_study()
    if args.check_inputs:
        print(json.dumps({"evidence_gate":"passed","executed_runs":len(study["summaries"]),
                          "reported_runs":study["plan"]["final_reported_runs"]}))
        return
    build_pdf(study,args.output.resolve())


if __name__ == "__main__":
    main()
