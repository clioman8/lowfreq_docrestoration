"""Restore omitted inputs from the user's original DB assets; no download or training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "experiment"
sys.path.insert(0, str(PROJECT))
from docrestore.assets import check_corpus, check_eng, prepare_fonts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--font-archive", type=Path, required=True)
    parser.add_argument("--eng", type=Path, required=True)
    args = parser.parse_args()
    check_corpus(args.corpus)
    check_eng(args.eng)
    selection = PROJECT / "assets/english_selection"
    manifest = json.loads((selection / "source_manifest.json").read_text(encoding="utf-8"))
    wanted = {row["archive_member"]: row for row in manifest["books"]}
    output = PROJECT / "assets/english_books"
    output.mkdir(parents=True, exist_ok=True)
    found = set()
    with tarfile.open(args.corpus, "r|gz") as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            row = wanted[member.name]
            if not member.isfile() or not str(row["book_id"]).isdigit():
                raise ValueError("Invalid selected corpus member")
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("Corpus member hash differs: " + member.name)
            (output / (str(row["book_id"]) + ".txt")).write_bytes(data)
            found.add(member.name)
            if len(found) == len(wanted):
                break
    if found != set(wanted):
        raise ValueError("Selected source books are missing")
    prepare_fonts(args.font_archive, PROJECT / "assets/fonts")
    target = PROJECT / "assets/eng.traineddata"
    if args.eng.resolve() != target.resolve():
        shutil.copyfile(args.eng, target)
    split = PROJECT / "data/development/split_manifest.json"
    split_source = PROJECT / "provenance/source_partition/development_split_manifest.json"
    split.parent.mkdir(parents=True, exist_ok=True)
    if split_source.resolve() != split.resolve():
        shutil.copyfile(split_source, split)
    print(json.dumps({"restored_source_books": len(found), "training_started": False}))


if __name__ == "__main__":
    main()
