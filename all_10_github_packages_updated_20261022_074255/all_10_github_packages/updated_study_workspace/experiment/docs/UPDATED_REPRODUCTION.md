# Updated code snapshot

Packaged 2026-10-22T07:42:55.665604+09:00. Snapshot kind: **code and reproducibility materials**.
Study state at packaging: **running** / B_lambda1_seed1.
Packaging basis: verified extracted original members. The original outer all_10_github_packages.zip is presently unavailable. The ten nested ZIPs were verified against PACKAGE_INDEX, and the original index/READ_FIRST bytes were verified against original_metadata in the SHA-verified immutable first snapshot. No original outer archive was reconstructed or verified as unchanged. The historical outer SHA, if recorded, is provenance only. A final bundle claiming all four unchanged original attachments still requires the exact missing original outer ZIP.

This ZIP contains no main training checkpoints, OCR result files, generated images,
final performance tables, or final thesis PDF. The original DB ZIP and design-reference
ZIP remain unchanged and are supplied separately. Research contents from the design
reference ZIP are not incorporated into this package.

The original ten nested ZIP names and common lowfreq_docrestoration/ layout are retained.
Current duplicate modules have identical bytes. Nested ZIPs remain stage-specific;
merge all ten for the full pipeline or use updated_study_workspace/ directly.

validation/*.json retains the original historical packaging/smoke records. These are
not current performance findings. validation/latest/ contains the separate merged-project
software record: 45 tests passed, including real CUDA backward and
exact CPU checkpoint-resume verification. It does not claim that every isolated nested
ZIP was independently re-tested, or that the main comparison has completed.

Current changes include metadata-described English source verification, ordered parallel
OCR (--workers), strict deterministic training, official Tesseract build version matching,
exact opt-in --resume with Adam/RNG/loader recovery, atomic checkpoints, explicit
training.eval_steps, Windows UTF8/font test portability, evidence-gated manuscript
generation, seed-wise secondary mean shifted CER reporting, and qualified contributor/work
group terminology. Original starter guides describe earlier example defaults; the frozen JSON
and this update guide take precedence for the archived study.

Frozen main configuration: 8,000/1,000/1,000 train/validation/test regions, 10,000 updates,
seeds 0/1/2, equal B/C lambda candidates [0.1,1,10], checkpoints at
[250,500,1000,2000,4000,6000,8000,10000], and 16 OCR workers. The protocol plans 13 completed
training runs and 9 selected reported repetitions; selected seed-0 tuning runs are reused.
These are protocol settings, not observed performance results.

See updated_study_workspace/REPRODUCTION_README.md, CHANGELOG.json,
CURRENT_SOURCE_SHA256.json and ARCHIVE_CONTENTS_SHA256.json for reproducibility and hashes.
