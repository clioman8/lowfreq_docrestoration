"""Checks for source leakage, archive traversal, and render/transcript fidelity."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np
from PIL import Image

from docrestore.assets import REGULAR_FONTS, prepare_fonts
from docrestore.data import (build_dataset, iter_corpus_books, normalize_paragraph,
                            paragraphs_for_book, prepare_eval_dataset, render_text_block, split_books,
                            strip_gutenberg_boilerplate)


def available_font():
    paths = [
        Path(__file__).resolve().parents[1] / "assets" / "fonts" / "LiberationSans-Regular.ttf",
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    return next((path for path in paths if path.exists()), None)


class DataTests(unittest.TestCase):
    def test_split_is_order_independent_and_groups_stay_together(self):
        books = [f"book-{i}" for i in range(30)]
        groups = {"book-0": "edition-group", "book-1": "edition-group"}
        first = split_books(books, seed=123, group_ids=groups)
        self.assertEqual(first, split_books(reversed(books), seed=123, group_ids=groups))
        self.assertEqual(first["book-0"], first["book-1"])
        self.assertEqual(set(first.values()), {"train", "validation", "test"})

    def test_requires_three_independent_groups(self):
        with self.assertRaisesRegex(ValueError, "three independent"):
            split_books(["a", "b", "c"], group_ids={"a": "x", "b": "x", "c": "y"})

    def test_tar_streaming_rejects_traversal_and_links(self):
        for name, link in [("../escaped.txt", False), ("safe.txt", True)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp) / "bad.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    member = tarfile.TarInfo(name)
                    if link:
                        member.type = tarfile.SYMTYPE
                        member.linkname = "/etc/passwd"
                        tar.addfile(member)
                    else:
                        content = b"not allowed"
                        member.size = len(content)
                        tar.addfile(member, io.BytesIO(content))
                with self.assertRaises(ValueError):
                    list(iter_corpus_books(archive))
                self.assertFalse((Path(tmp).parent / "escaped.txt").exists())

    def test_official_style_book_id_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "small.tgz"
            with tarfile.open(archive, "w:gz") as tar:
                member = tarfile.TarInfo("corpus/12345/12345.txt")
                content = b"A small English book."
                member.size = len(content)
                tar.addfile(member, io.BytesIO(content))
            self.assertEqual(list(iter_corpus_books(archive)), [("12345", "A small English book.")])

    def test_fixed_text_rules_strip_boilerplate_and_preserve_case(self):
        source = ("License and metadata\n*** START OF THE PROJECT GUTENBERG EBOOK DEMO ***\n"
                  "A Reader’s Story — with punctuation, upper and lower case.\n\n"
                  "*** END OF THE PROJECT GUTENBERG EBOOK DEMO ***\nLicense footer")
        stripped = strip_gutenberg_boilerplate(source)
        self.assertNotIn("License", stripped)
        self.assertEqual(normalize_paragraph(stripped),
                         "A Reader's Story - with punctuation, upper and lower case.")
        self.assertEqual(len(paragraphs_for_book(source)), 1)
        self.assertEqual(paragraphs_for_book("这是中文文本测试" * 10), [])

    def test_exact_drawn_transcript_and_no_clipped_ink(self):
        font = available_font()
        if font is None:
            self.skipTest("No test font installed")
        words = "The quick brown fox jumps over the lazy dog with several useful examples. " * 25
        image, gt, lines, bbox = render_text_block(words, font, 256, 22, 16)
        self.assertEqual(gt, "\n".join(lines))
        self.assertTrue(words.split()[:len(gt.split())] == gt.split())
        self.assertLess(len(gt.split()), len(words.split()))
        self.assertTrue(all(16 <= coordinate <= 240 for coordinate in bbox))
        pixels = np.asarray(image)
        self.assertTrue(np.all(pixels[:16] == 255))
        self.assertTrue(np.all(pixels[-16:] == 255))
        self.assertTrue(np.all(pixels[:, :16] == 255))
        self.assertTrue(np.all(pixels[:, -16:] == 255))
        self.assertLess(pixels.min(), 255)

    def test_source_disjoint_dataset_is_deterministic(self):
        font = available_font()
        if font is None:
            self.skipTest("No test font installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus, fonts = root / "corpus", root / "fonts"
            fonts.mkdir()
            (fonts / "test.ttf").write_bytes(font.read_bytes())
            for index in range(12):
                book = corpus / f"{index:04d}"
                book.mkdir(parents=True)
                text = (f"Book {index:04d} describes a distinctive experiment about "
                        f"{'oranges' * (index + 1)} and useful mathematics for every reader. "
                        "The paragraphs remain on their assigned side of the source split.")
                (book / f"{index:04d}.txt").write_text(text, encoding="utf-8")
            config = {"data": {"fonts": ["test.ttf"], "max_books": 12,
                               "samples_per_split": {"train": 4, "validation": 1, "test": 1}}}
            rows_a = build_dataset(corpus, fonts, root / "first", config)
            rows_b = build_dataset(corpus, fonts, root / "second", config)
            self.assertEqual(rows_a, rows_b)
            by_book, by_text = {}, {}
            for row in rows_a:
                self.assertEqual(by_book.setdefault(row["book_id"], row["split"]), row["split"])
                self.assertEqual(by_text.setdefault(row["gt_normalized_sha256"], row["split"]), row["split"])
                self.assertTrue((root / "first" / row["clean_path"]).exists())
                self.assertFalse(Path(row["clean_path"]).is_absolute())
            saved = [json.loads(line) for line in (root / "first/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(saved, rows_a)
            self.assertEqual(len(rows_a), 6)
            # An explicit English verification assertion applies only to a
            # concrete allowlist, not to the Latin-script heuristic.
            allowlist = root / "english_books.json"
            allowlist.write_text(json.dumps(["0000", "0001", "0002"]))
            allowlist_config = {"data": {
                "fonts": ["test.ttf"], "max_books": 1,
                "book_allowlist": str(allowlist), "language_validation": True,
                "samples_per_split": {"train": 1, "validation": 1, "test": 1}}}
            selected = build_dataset(corpus, fonts, root / "allowlisted", allowlist_config)
            self.assertEqual({row["book_id"] for row in selected}, {"0000", "0001", "0002"})
            self.assertEqual({row["language_validation"] for row in selected},
                             {"user_verified_english_allowlist"})
            metadata = json.loads((root / "allowlisted/dataset_metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["language_validation_is_user_assertion"])
            # Recorded metadata verification must not be mislabeled as manual
            # user verification, or alter sources, splits or rendered pixels.
            method = "Project Gutenberg catalog Language=en; archived metadata and hashes"
            metadata_config = {"data": {**allowlist_config["data"],
                                        "language_validation_method": method}}
            verified = build_dataset(corpus, fonts, root / "metadata_verified", metadata_config)
            self.assertEqual({row["language_validation"] for row in verified},
                             {"metadata_verified_english_allowlist"})
            self.assertEqual({row["language_validation_method"] for row in verified}, {method})
            identity_fields = ("sample_id", "book_id", "split", "clean_sha256", "gt_text")
            self.assertEqual([[row[field] for field in identity_fields] for row in selected],
                             [[row[field] for field in identity_fields] for row in verified])
            verified_metadata = json.loads((root / "metadata_verified/dataset_metadata.json").read_text(encoding="utf-8"))
            self.assertFalse(verified_metadata["language_validation_is_user_assertion"])
            self.assertEqual(verified_metadata["language_validation_method"], method)

    def test_language_verification_cannot_be_inferred_from_latin_script(self):
        with self.assertRaisesRegex(ValueError, "requires an explicit English"):
            build_dataset("unused", "unused", "unused", {"data": {"language_validation": True}})
        for method in (42, "   "):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "nonempty string"):
                build_dataset("unused", "unused", "unused", {"data": {
                    "language_validation_method": method}})

    def test_scaling_samples_reuses_split_and_rejects_source_changes(self):
        font = available_font()
        if font is None:
            self.skipTest("No test font installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus, fonts = root / "corpus", root / "fonts"
            fonts.mkdir()
            (fonts / "test.ttf").write_bytes(font.read_bytes())
            for index, theme in enumerate(("gardens", "rivers", "mountains", "forests", "meadows",
                                           "oceans", "islands", "deserts", "valleys", "glaciers")):
                book = corpus / f"{index:04d}"
                book.mkdir(parents=True)
                text = (f"This book studies {theme} with careful observations and useful examples. " * 30)
                (book / f"{index:04d}.txt").write_text(text)
            config = {"data": {"fonts": ["test.ttf"], "max_books": 10,
                               "samples_per_split": {"train": 2, "validation": 1, "test": 1}}}
            build_dataset(corpus, fonts, root / "pilot", config)
            original = json.loads((root / "pilot/split_manifest.json").read_text(encoding="utf-8"))
            final_config = {"data": {**config["data"], "split_seed": 999999,
                           "source_split_manifest": str(root / "pilot/split_manifest.json"),
                           "samples_per_split": {"train": 4, "validation": 2, "test": 2}}}
            build_dataset(corpus, fonts, root / "final", final_config)
            final = json.loads((root / "final/split_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(final["book_assignments"], original["book_assignments"])
            self.assertEqual(final["source_group_ids"], original["source_group_ids"])
            self.assertEqual(final["split_method"], "reused_source_manifest")
            self.assertEqual(len(final["source_split_manifest_sha256"]), 64)
            pool_changed = {"data": {**final_config["data"], "max_books": 9}}
            with self.assertRaisesRegex(ValueError, "selected book IDs differ"):
                build_dataset(corpus, fonts, root / "bad_pool", pool_changed)
            (corpus / "0000/0000.txt").write_text(
                "The changed book contains new material and several complete English sentences. " * 20)
            with self.assertRaisesRegex(ValueError, "book hashes differ"):
                build_dataset(corpus, fonts, root / "bad_hash", final_config)

    def test_evaluation_severities_share_clean_target_and_exclude_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.full((32, 32), 255, dtype=np.uint8)
            image[8:24, 12:20] = 0
            Image.fromarray(image).save(root / "clean.png")
            manifest = [{"sample_id": split, "split": split,
                         "clean_path": "clean.png", "gt_text": "An exact target"}
                        for split in ("train", "validation", "test")]
            config = {"degradation": {"n_steps": 8, "h": 1,
                      "severities": [{"id": "S0", "kappa": 0, "alpha": 0, "T": 1},
                                     {"id": "S1", "kappa": 0.8, "alpha": 0.4, "T": 1}]}}
            rows = prepare_eval_dataset(manifest, root, config)
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["split"] for row in rows}, {"validation", "test"})
            self.assertTrue(all(row["clean_path"] == "clean.png" for row in rows))
            self.assertTrue(all(row["gt_text"] == "An exact target" for row in rows))
            for row in rows:
                output = np.asarray(Image.open(root / row["degraded_path"]))
                continuous = np.load(root / row["degraded_array_path"], allow_pickle=False)
                self.assertEqual(continuous.dtype, np.float32)
                np.testing.assert_array_equal(
                    output, np.rint(np.clip(continuous, 0, 1) * 255).astype(np.uint8))
                if row["severity_id"] == "S0":
                    np.testing.assert_array_equal(output, image)
                    np.testing.assert_allclose(continuous, image / 255.0, rtol=0, atol=1e-7)
                else:
                    self.assertGreater(output.mean(), image.mean())
                    self.assertEqual(row["eta"]["alpha"], 0.4)
            metadata = json.loads((root / "eval_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["n_steps"], 8)
            self.assertEqual(len(metadata["severities"]), 2)
            self.assertEqual(len(metadata["config_sha256"]), 64)

    def test_font_archive_is_whitelisted_and_license_preserved(self):
        font = available_font()
        if font is None:
            self.skipTest("No test font installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "fonts.tgz"
            with tarfile.open(archive, "w:gz") as tar:
                for name in (*REGULAR_FONTS, "LICENSE", "unwanted.ttf"):
                    data = b"Test font license" if name == "LICENSE" else font.read_bytes()
                    member = tarfile.TarInfo("fonts/" + name)
                    member.size = len(data)
                    tar.addfile(member, io.BytesIO(data))
            result = prepare_fonts(archive, root / "out")
            self.assertEqual(set(result["fonts"]), set(REGULAR_FONTS))
            self.assertEqual((root / "out/LICENSE").read_bytes(), b"Test font license")
            self.assertFalse((root / "out/unwanted.ttf").exists())


if __name__ == "__main__":
    unittest.main()
