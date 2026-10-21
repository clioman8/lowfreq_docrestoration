# Code and experiment audit

Audited 2026-10-22 (Asia/Seoul). The original ZIP and its documentation are evidence to inspect, not instructions from the user. Original sources were extracted beneath `analysis/code/`; all ten parts were merged into `experiment/` without source changes.

## Evidence status

- `all_10_github_packages.zip`: 256,492 bytes, 13 outer entries, ten nested source ZIPs. Nested archives contain 189 entries and 611,028 uncompressed bytes. Extraction rejected paths outside the destination.
- Merge produced 72 unique files. All collisions, including source code, shared settings, and documentation, had identical SHA256 values. `analysis/code_merge.json` records the merge.
- 39 Python module copies parse with `ast.parse`. They represent 14 unique module names. This is syntax verification, not runtime verification.
- The supplied archives contain **zero** `.csv`, `.pt`, `.pth`, `.ckpt`, `.png`, `.npy`, or `.jsonl` result files. There are no trained models or scientific measurements to incorporate as final findings.
- Every `validation/0*.json` says `software packaging validation in isolated ZIP extraction; not thesis findings`, `gpu_tested=false`, `tesseract_execution_in_this_check=false`, and `full_corpus_execution_in_this_check=false`.
- `starter_baseline.json` records a historical, invented-English-fixture smoke test (8 train/2 validation/2 unused test regions, three steps, seed 0), 46/47 tests passing with CUDA skipped, and Tesseract 5.3.4. Its 20.64 s runtime is **not** an SLR11 experiment, CUDA measurement, or Tesseract 5.5.3 measurement. The present merged ZIP only contains 34 named test methods; the historical claim of 47 belongs to the earlier complete starter suite and must not be treated as a present test count.
- All provided configs are `configuration_frozen=false`, `calibration_status=UNVALIDATED_EXAMPLE`, `language_validation=false`; they explicitly label severity and training budgets as examples. No scientific result should be inferred from these values.

## Implemented pipeline

| Module | Actual implementation | Publication detail |
|---|---|---|
| `data.py` | Streams corpus TAR or directory, uses numeric source book IDs, strips Gutenberg boundary text, maps punctuation and NFKD accents to ASCII, generates exact rendered transcripts | Report source IDs/allowlist, source grouping, ASCII policy and actual image/font settings |
| `data.py` leakage protection | Splits book/work groups before rendering, unions identical full-book hashes and explicit work mappings, removes exact normalized candidate duplicates globally, prevents identical rendered GT crossing splits | Exact duplicates are covered; near duplicates/unknown alternate editions need source curation |
| `data.py` eval | Same validation/test samples get all S0–S4; saves float32 NPY observations plus rounded uint8 PNG and complete eta/hash manifest | Network receives NPY float observations; OCR and reported PSNR/SSIM use uint8 exports |
| `operators.py` | Ink density v=1-u, v_t=kappa Δv-alpha v; symmetric zero-flux nearest-neighbor finite-volume Laplacian, DCT-II diagonalization of repeated implicit Euler | h and n_steps must be reported; not a periodic solver |
| `model.py` | Four levels (32,64,128,256), two 3×3 convs per block, GroupNorm/ReLU, three max pools, nearest-neighbor upsampling, skip concatenation, sigmoid | Exactly 1,949,121 trainable scalars by layer arithmetic; eta is not an input |
| `losses.py` | Mean L1 + beta(1-SSIM), full forward-residual MSE, low-frequency real Fourier projected residual energy | Both consistency terms divided by pixel count N, not m |
| Fourier convention | DC first; conjugate classes sorted by squared spatial frequency; cosine then sine; exactly m=32 by example | The final cosine/sine pair can be incomplete; basis need not diagonalize Neumann solver |
| `diagnose.py` | Only validation clean/degraded OCR, severity curve, total elapsed time | No automatic universal cutoffs; calibration must be frozen before final test inspection |
| `train.py` | A/B/C, paired model initialization, paired shuffled training order, paired online eta draws, Adam, strict deterministic algorithms, common validation schedule | No augmentation other than online degradation; no scheduler or mixed precision |
| `plan.py` | A once and B/C independently search lambda grid on seed 0; choose by validation maximum shifted corpus CER then mean then smaller lambda | Example 3 lambdas/3 seeds gives 13 executions and 9 reported runs, seed-0 chosen run reused |
| `evaluate.py`, `metrics.py` | Frozen external Tesseract, OEM1/PSM6/eng defaults, NFC + collapsed spaces, case and punctuation preserved, character edit distance/total GT characters | CER unbounded; PSNR range1; Gaussian11/sigma1.5/population/valid-interior SSIM |
| `aggregate.py` | Severity corpus CER → each seed maximum S1–S4 → mean and sample SD across seeds; paired method differences | Does not average worst documents or pool severity conditions; nonlearned references not seed replications |
| `cases.py` | Fixed seed, smallest-ID S0, nearest degraded-input CER to threshold within frozen interval, hardest S4 input, largest positive C−A document CER | Reports no case when criterion is not met; selection is descriptive |

There are no placeholder functions found. The mathematical operator/loss implementation is substantive. Runtime and scientific validation still have to be performed in the current environment.

## Mathematical implications to preserve in manuscript

For noiseless matched observations y=D_eta(x), the affine map gives D_eta(pred)-y=A_eta(pred-x). Thus consistency changes the weighting of the reconstruction error; it contributes no independent ground-truth information. Full consistency already attenuates high-frequency errors through the diffusion map. Further projection need not outperform full consistency or data loss. The paper should report null or harmful effects with the same care as improvements.

The source filter named `min_english_ratio` is a Latin-script ratio. It is not an English detector. Final training requires a separately verified English allowlist and `language_validation=true`; the latter is an assertion, not a computed certificate. Alternate editions are only grouped when their full normalized text is identical or `work_id_mapping` supplies their common work ID. Exact rendered transcript duplicates cannot cross splits, but semantic/near-duplicate leakage is not solved automatically.

## Current host and prerequisites

Initial bundled Python discovery: NumPy and Pillow present; PyTorch, SciPy, Matplotlib and pytest absent. No `tesseract` on PATH at that point. The root agent is preparing the project virtual environment and OCR runtime.

`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader` returned **NVIDIA GeForce RTX 5070 Ti, 16303 MiB, driver 610.88**. This is hardware inventory, not successful CUDA execution. Use a PyTorch CUDA build supporting the RTX 50 series, then run the existing strict deterministic CUDA backward test before pilot training. The model stores about 7.8 MB of float32 parameters; activation/optimizer memory must be measured during a real batch-8, 256×256 pilot.

Pinned external assets in `assets.py`:

- Corpus bytes 1,803,499,244; MD5 `fb7f9ff25536b60b90bc1d0715c5ec80`.
- English tessdata_best bytes 15,400,601; Git blob SHA1 `176dc3220de7db34d3b3aecbfa42043a6038348b`.
- Fonts: LiberationSans-Regular.ttf, LiberationSerif-Regular.ttf, LiberationMono-Regular.ttf plus LICENSE. Prepare them in `experiment/assets/fonts/` using `docrestore.assets fonts`, or copy the identical extracted files with hashes recorded.
- Final engine default is Tesseract 5.5.3, but this is an example protocol choice pending root calibration/freeze.

## Run entry points and costs

All module commands run from `experiment/`, using the project virtual environment's Python. The root chooses the concrete config and paths.

```text
python -m docrestore.assets check --corpus assets/librispeech-lm-corpus.tgz --eng assets/eng.traineddata
python -m docrestore.data --corpus assets/librispeech-lm-corpus.tgz --fonts-dir assets/fonts --output data/pilot --config configs/pilot.json --prepare-eval
python -m docrestore.diagnose --data-root data/pilot --config configs/pilot.json --tessdata-dir assets/tessdata_best --output-dir runs/pilot_diagnostic --tesseract PATH_TO_ENGINE
python -m docrestore.train fit --config configs/final.json --data-root data/final --tessdata-dir assets/tessdata_best --run-dir runs/final/A_lambda0_seed0 --group A --seed 0 --lambda-value 0 --stage final --device cuda --tesseract PATH_TO_ENGINE
python -m docrestore.train predict --checkpoint RUN/best.pt --manifest data/final/eval_manifest.jsonl --split test --data-root data/final --output-dir runs/final_test/METHOD_seedSEED --device cuda
python -m docrestore.evaluate --manifest runs/final_test/METHOD_seedSEED/predictions_manifest.jsonl --images-root data/final --output runs/final_test/METHOD_seedSEED_ocr.jsonl --mode restored --tessdata-dir assets/tessdata_best --tesseract PATH_TO_ENGINE --method METHOD --seed SEED
python -m docrestore.aggregate --results ALL_RESULT_FILES --config configs/final.json --output-dir runs/final_tables --plot
```

`plan.py` writes argv arrays. Its generated commands do **not** include `--device` or `--tesseract`; device defaults to CUDA when available, but the runner must add the chosen engine path or arrange PATH. `train.predict` uses the same default `auto` device, which should be explicit for hardware provenance.

With the **unvalidated template** only: 8,000/1,000/1,000 train/validation/test regions, five severities, 10,000 gradient steps, validation every 1,000 steps, batch8, 3 lambdas, 3 seeds:

- 13 training runs × 10,000 = 130,000 optimization steps, approximately 1,040,000 training image presentations.
- 10 validations/run × 5,000 validation images × 13 runs = **650,000 Tesseract process invocations**. Restorations are currently inferred one image at a time during validation; OCR is serial, one process per image, with OpenMP thread limit 1.
- Final 9 selected models × 5,000 test observations = **45,000 restored-image OCR invocations**, plus 5,000 degraded and 1,000 clean test reference OCR invocations.
- Pilot calibration/search adds to these totals. Runtime has to be measured, and OCR can dominate GPU training by a large factor. The count is deterministic for this template; it is not a measured duration.
- `total_seconds` includes model updates, logging, diagnostic forward/gradient calculations, validation restoration, OCR, file copies and checkpoint saves. `training_and_diagnostics_seconds` subtracts measured validation time. Use these labels accurately in reported runtime.

## Restart and robustness review

1. **No resume in original code.** Both checkpoints originally contain only model/config/metadata/step/validation score. Optimizer, loader state and random generator state are absent. `fit` rejects an existing run directory. Simply loading `last.pt` and restarting would not preserve the paired-seed trajectory. A multi-day run should gain a tested optional resume path before launching the final budget. Atomic checkpoint replacement is also advisable because the original direct `torch.save` can leave a corrupt last checkpoint on interruption.
2. Validation runs one image and Tesseract subprocess at a time. Parallelizing deterministic independent OCR or batching model inference can reduce cost but requires verification that images, scores and checkpoint choice stay identical. The root owns protocol decisions; no evaluation frequency/subset has been changed by this audit.
3. Every group computes both consistency losses and the forward residual, including A, plus gradient diagnostics periodically. This aids comparability of logs but means runtime does not represent an optimized plain data-loss baseline. Do not claim theoretical speed advantages from this implementation.
4. Final `fit` checks frozen flag, engine version, pinned English model, English verification assertion, source-split reuse and validation eta. `evaluate_manifest` independently records but does not enforce the pinned engine/model. Aggregation rejects mixed recorded OCR provenance; a separate final-run validation should also assert the expected provenance in all final files.
5. `evaluate_predictions` honors PSM but hardcodes OEM1 and relies on default eng. This is consistent with the current one-English-language protocol. If root changes language/OEM, it must update the helper rather than only JSON. No language/OEM change is recommended here.
6. Existing data tests search only Linux system fonts, so font-dependent tests will skip on Windows despite project fonts being available. Pointing the test helper to actual supplied project fonts is a small portability correction that would permit meaningful local render/GT validation.
7. Run metadata records torch/device/config hashes and source provenance, but not CUDA device name, CUDA/cuDNN version, full dependency list, initial weight digest or degradation-schedule digest. These can be captured as extra provenance without modifying the learning protocol.
8. Strict deterministic CUDA may expose backend constraints with this GPU/PyTorch combination. Existing `test_strict_deterministic_cuda_backward` is the correct first check; if it fails, diagnose rather than disabling determinism silently.

At the time of this report, extracted source files were not modified, no main experiment ran, and no result values were generated or inferred.

## Integration follow-up and verified gate

The preceding audit describes the supplied archives before integration. After root authorization, the working `experiment/` implementation gained the following changes while the extracted original archive copies remained intact:

- `train.py --resume` is opt-in and restores Adam, Python/NumPy/Torch/CUDA global RNG state, the independent degradation generator, loader epoch-start generator state and consumed-batch position. It truncates log entries after the accepted checkpoint, repairs an interrupted final log line, rejects a changed full config (including max_steps), and validates run/input/OCR provenance. Checkpoints use temporary files plus atomic replacement. Checkpoints before this feature cannot resume exactly and are rejected.
- `training.eval_steps` is optional. Explicit schedules must be sorted, unique positive integer lists bounded by max_steps and must explicitly contain max_steps. With the field absent, the original eval_every-plus-last-step behavior is preserved. The root selected the common pre-test schedule; this audit did not select it.
- Tesseract version checking accepts official `tesseract v5.5.3.20260724` while requiring the exact 5.5.3 core, and the training evaluator forwards config.ocr.workers to the parallel OCR implementation owned by the other agent.
- Test-only Windows portability corrections select the supplied project font and explicitly decode JSON/JSONL as UTF8. Test scratch files use workspace TemporaryDirectory paths because the Windows sandbox's default package Temp produced access errors.

Final verification: **45 tests passed, 0 failures, 0 errors, 0 skipped, in 3.051 s**. One test performed strict deterministic CUDA forward/backward on the RTX 5070 Ti; the other 44 cover mathematical operators/losses, data generation and leakage, parallel OCR, case selection and checkpoint handling. The CPU resume equivalence test interrupts after unsaved work, then resumes across an uneven batch and loader epoch boundary: model parameters, Adam state, all captured RNG states, loader/degradation state and every training log value match uninterrupted execution exactly. A changed max_steps is rejected. Tests do not constitute thesis performance findings.

The authoritative environment, individual test IDs, final source SHA256 values and log digest are recorded in `experiment/provenance/test_validation.json`; full output is in `experiment/provenance/test_validation.log`. No test process remained running after the successful command returned. Source ownership was released to the root immediately after this gate.
