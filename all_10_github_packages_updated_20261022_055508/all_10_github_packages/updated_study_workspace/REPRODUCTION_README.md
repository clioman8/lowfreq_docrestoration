# Reproduction workspace — code snapshot

Snapshot: 2026-10-22T05:55:08.533731+09:00; study state: running /
C_lambda1_seed0. No final experimental findings or completed PDF are in this code ZIP.
The completion handoff is an archival record: its process IDs, session IDs, automation status,
absolute paths and heartbeat timestamps are not live instructions on another machine.

This folder mirrors the original workspace: scripts/, analysis/, experiment/. Main code,
all five test modules, the actual frozen configuration, source selection metadata, source
partition metadata and current implementation/data-integrity checks are included. Heavy input
   assets, Tesseract installation, .venv, all generated image/observation data, active process state, runs, results,
predictions, images, caches and final PDF are deliberately omitted. source_partition/ contains
only the pre-existing book-group split metadata, not rendered observations or OCR results.
The single experiment/data/development/split_manifest.json file is also included at the exact
relative path referenced by frozen.json; it contains source-group assignments, not image data.

On Windows, from this folder:

1. Create .venv with Python 3.13 and install experiment/requirements.txt. The recorded tested
   environment is experiment/provenance/environment_requirements.txt; the tested GPU build is
   PyTorch 2.10.0+cu128 on RTX 5070 Ti. Select a compatible CUDA build for the target hardware.
   PDF creation additionally needs reportlab and pypdf; matplotlib and Pillow are already listed.
2. Extract the user's unchanged DB ZIP outside this folder. Prepare its corpus TGZ, Liberation
   2.1.5 font TAR.GZ and eng.traineddata using the included input-preparation helper:

   .venv\Scripts\python.exe scripts\prepare_supplied_inputs.py --corpus CORPUS_TGZ --font-archive FONT_TAR_GZ --eng ENG_TRAINEDDATA

   It verifies official corpus/model identifiers, restores exactly the 500 selected source texts,
   preserves the three Regular font files and license, and restores the frozen source split.
   It does not render images, train, inspect OCR or change scientific code/configuration.
3. Place the full Tesseract 5.5.3 Windows installation (including its DLLs) under
   experiment/tools/tesseract/, so tools/tesseract/tesseract.exe exists relative to experiment/.
   The tested official build was v5.5.3.20260724. Keep the archived eng.traineddata fixed.
4. Preserve configs/frozen.json and provenance/study_lock.json. Do not run freeze_study.py
   over the archived frozen configuration. prepare_study.py and freeze_study.py are retained
   for provenance and for a separately designed study, not for changing this frozen one.
5. For tests, run from experiment/ with the virtual environment Python. On the restricted
   Windows sandbox used here, set tempfile.tempdir to a workspace-owned scratch folder first.
   The archived test_validation record describes the full merged suite, not isolated packages.
6. To reproduce the frozen main pipeline from this folder, after supplying omitted inputs:

   .venv\Scripts\python.exe -u scripts\run_study.py

   It checks frozen hashes, generates main observations, performs validation-only diagnostics,
   trains seven seed-0 candidates then six selected-seed runs, and only then evaluates held-out
   test documents and generates aggregate/case records. Do not launch a duplicate pipeline
   against an already running original workspace. The code snapshot contains no active status
   file or checkpoints and cannot resume the original machine's in-progress execution.
7. After all results exist, build_manuscript.py --check-inputs verifies evidence. PDF authoring
   also needs the complete results and Times fonts, then every PDF page must be rendered and
   visually reviewed. The included analysis/manuscript_sections.json contains methodological
   prose; empirical abstract/results/discussion/conclusion are generated from completed records.

Frozen absolute paths recorded inside historical provenance identify the original machine.
The runnable scripts derive their workspace from __file__; the restored source-partition file
is copied to the relative path referenced by frozen.json. Archived generated-data hashes refer
to the original study and are evidence, not files supplied in this code-only package.

The ten stage-specific ZIPs can instead be merged into lowfreq_docrestoration/. All duplicated
modules match the current frozen version. Original historical validation objects are retained
byte-for-byte; current records live under validation/latest/ or experiment/provenance/.
The final delivery bundle is assembled separately by scripts/package_final_delivery.py when
available; it may add the unchanged original four attachments, reviewed PDF and final evidence.
