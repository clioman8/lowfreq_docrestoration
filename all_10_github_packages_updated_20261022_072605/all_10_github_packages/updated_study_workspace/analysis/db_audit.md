# Supplied DB asset audit

Audit date (client timezone Asia/Seoul): 2026-10-22. This audit treats all supplied files as evidence rather than instructions. It does not contain restoration or OCR experiment results.

## Archive inventory and actual data status

The supplied DB ZIP is 1,819,223,408 bytes and contains one directory plus three files:

| File | Uncompressed ZIP-member bytes | Status |
| --- | ---: | --- |
| `eng.traineddata` | 15,400,601 | English pretrained OCR language asset; not a restoration checkpoint |
| `liberation-fonts-ttf-2.1.5.tar.gz` | 2,385,008 | Liberation font archive |
| `librispeech-lm-corpus.tgz` | 1,803,499,244 | Project Gutenberg source text corpus |

There are no supplied clean/degraded document images, rendered ground-truth pairs, source split manifests, severity parameter samples, A/B/C restoration model checkpoints, learning histories, measured CER/PSNR/SSIM results, aggregates, or selected failure cases in this DB archive. These must be produced by executing the experiment. The supplied asset itself cannot substantiate numerical superiority of any model or loss.

The corpus TGZ has 28,956 tar members, of which 14,478 are regular `.txt` files: **14,476 source books** and two metadata files (`BOOKS.TXT`, 1,165,569 bytes; `README.TXT`, 749 bytes). The source books total **4,686,244,335 bytes**. All source books decoded strictly as UTF-8. Book paths are `librispeech-lm-corpus/corpus/<numeric-ID>/<numeric-ID>.txt` and preserve Project Gutenberg IDs.

The full corpus stream was inspected without extracting all books. The initial comprehensive scan took 331.828 seconds, including per-book byte hashes, decoded text statistics, whitespace/casefold hashes, and metadata inventories. No exact byte-hash duplicate books and no whitespace/casefold full-book duplicates were found. This does **not** establish an absence of partial overlap or alternative editions.

`README.TXT` says that the source is Project Gutenberg; that Gutenberg headers and footers were removed for language modelling; and that source selection attempted to avoid even partial overlap with books used in LibriSpeech's training and development audio data. That source selection is unrelated to the present restoration experiment's train/validation/test split. `BOOKS.TXT` has IDs/titles/authors but often lacks authors and can contain multiline titles, so it was not used as the sole language or author verification source.

## Verified asset identities and provenance

| Asset | MD5 | SHA256 |
| --- | --- | --- |
| Corpus TGZ | `fb7f9ff25536b60b90bc1d0715c5ec80` | `9d0da0460cf7b1e7413ff8ed5ba4f8f14ae6bee0b3ffb2842c5d234515bd77c9` |
| English traineddata | `4be3f51b55c0074d8c6b1ee5b5100f95` | `8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba` |
| Font TGZ | `31b453e0b77bacde410a34a725b34f8a` | `7191c669bf38899f73a2094ed00f7b800553364f90e2637010a69c0e268f25d0` |

The corpus MD5 and byte length match the official-source identity expected in the supplied audited code. [OpenSLR SLR11](https://us.openslr.org/11/) describes the resource as approximately 14,500 public-domain books used for LibriSpeech language modelling. Report the measured count of 14,476 for this archive. Do not confuse SLR11 text with the SLR12 speech corpus and its separate license.

The font archive contains twelve TTFs (Serif/Sans/Mono × Regular/Bold/Italic/BoldItalic) and README/LICENSE/AUTHORS/ChangeLog/TODO. The included license states SIL Open Font License 1.1; copyright notices name Google and Red Hat. The intended regular fonts and unchanged license were copied to `experiment/assets/fonts/`.

The supplied English traineddata is an LSTM-only asset: its traineddata directory has components 17–23 and no legacy OCR components. Its embedded version string is `4.00.00alpha:eng:synth20170629:[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]`. This supports the fixed `--oem 1` setting. Size and format are consistent with the official `tessdata_best` English model; the exact repository commit was not independently proven by this local audit. The [official tessdata documentation](https://tesseract-ocr.github.io/tessdoc/Data-Files.html) documents LSTM-only operation, and the [tessdata_best README](https://github.com/tesseract-ocr/tessdata_best/blob/main/README.md) states Apache-2.0 licensing. The supplied ZIP itself does not include that model's separate license file or a download manifest.

## Frozen English selection before image generation

The official [Project Gutenberg machine-readable catalog](https://dev.gutenberg.org/ebooks/offline_catalogs.html) was downloaded from `https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz` on client date 2026-10-22. The snapshot is 5,594,864 bytes and SHA256 `27299ac657fc8812aa11a108393abbe40c9e9d8e2c43b80229d23f31250f2aa8`; it has 79,634 records. Raw snapshot and provenance are saved in `analysis/db/metadata/`.

Every one of the 14,476 supplied source book IDs matched this catalog. Metadata language is exactly `en` for 14,475 and `en; zh` for one. For final candidate eligibility, the frozen metadata rule requires `Type=Text`, `Language=en`, exactly one nonempty semicolon-delimited named catalog contributor, excludes Anonymous/Unknown/Various, and enforces the reader's 16 MiB per-source limit. This leaves **9,899 eligible sources**. The **first 500 eligible books in original tar order** were selected. No clean/degraded image, OCR output, restoration model output, or test metric was used for source selection.

The fixed 500-book pool totals **210,728,223 bytes** and spans **210 named-author groups**. Each selected author's books are assigned the same `work_id_mapping` group; this is a conservative **author/work-group-disjoint** split strengthening the basic book-disjoint design. Two pairs have exactly matching catalog title/author despite distinct source bytes: IDs 73/463 (*The Red Badge of Courage*) and 219/526 (*Heart of Darkness*). The author grouping keeps each pair together. A title-only matching rule would not generally resolve collected works and partial inclusions. Multi-contributor/anonymous ambiguity was excluded before selection.

All 500 selected originals were extracted, unchanged, to `experiment/assets/english_books/<ID>.txt`. Every written file's SHA256 was checked against its original tar member. Source/archive hashes, member paths, output paths and source bytes are recorded in `experiment/assets/english_selection/source_manifest.json`. Extraction plus complete tail metadata traversal took 14.859 seconds; repeated dataset preparation can now read the 500-file directory directly.

For execution with working directory `experiment`:

```json
{
  "corpus": "assets/english_books",
  "data": {
    "book_allowlist": "assets/english_selection/book_allowlist.json",
    "language_validation": true,
    "work_id_mapping": "assets/english_selection/work_id_mapping.json"
  }
}
```

`corpus` is shown as an orchestration path; the provided data CLI takes it as `--corpus`. The actual data config keys are the three nested fields. Pilot and final rendering must reuse the same source partition manifest after the pilot split is created. Language verification is based on official catalog metadata and is distinct from the data code's Latin-script heuristic.

## Execution feasibility and remaining evidence

C: had approximately 1.887 TB free before asset extraction. All three supplied assets are now under `experiment/assets/`. There is ample source text for the example 8,000/1,000/1,000 rendering budget. Actual paragraph eligibility, the rendered text distribution, clean OCR quality, severity calibration, operator residual/gradient checks, timing, training convergence, checkpoint selection, and final test metrics still require their designated execution stages. None may be inferred from corpus quantity.

The whole-corpus scan's no-exact-duplicate finding is weaker than the fixed code's canonical full-text normalization check. The fixed data code also removes identical candidate chunks globally and refuses identical rendered ground-truth text across partitions. The author grouping helps with editions and collected works, but residual semantic similarity cannot be ruled out by SHA hashes alone. This is a limit to describe accurately, without claiming perfect semantic deduplication.

Machine-readable supporting files: `analysis/db/db_summary.json`, `corpus_inventory.json`, `corpus_file_audit.json`, `corpus_catalog_join.json`, `asset_hashes.json`, `english_pool_selection_summary.json`, and the self-contained `experiment/assets/english_selection/` manifests. Reproducible audit scripts are `analysis/db/audit_db.py`, `catalog_join.py`, `extract_english_pool.py`, and `check_pool_eligibility.py`.
