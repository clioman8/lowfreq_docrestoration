"""Prepare local, provenance-checked assets without downloading a corpus.

Only the three Liberation Regular fonts and their license are copied out of a
font archive.  No tar member is ever extracted by its archived path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile


REGULAR_FONTS = (
    "LiberationSans-Regular.ttf",
    "LiberationSerif-Regular.ttf",
    "LiberationMono-Regular.ttf",
)
CORPUS_MD5 = "fb7f9ff25536b60b90bc1d0715c5ec80"
CORPUS_BYTES = 1803499244
ENG_GIT_BLOB_SHA1 = "176dc3220de7db34d3b3aecbfa42043a6038348b"
ENG_BYTES = 15400601


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member_name(name: str) -> PurePosixPath:
    """Validate both POSIX and Windows traversal spellings."""
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or "\\" in name
            or (path.parts and ":" in path.parts[0])):
        raise ValueError(f"Unsafe archive member: {name!r}")
    return path


def prepare_fonts(archive: str | Path, output: str | Path) -> dict:
    output = Path(output)
    wanted = set(REGULAR_FONTS) | {"LICENSE"}
    selected: dict[str, bytes] = {}
    with tarfile.open(archive, "r:*") as members:
        for member in members:
            path = safe_member_name(member.name)
            if member.issym() or member.islnk():
                raise ValueError(f"Archive links are not allowed: {member.name}")
            if not member.isfile() or path.name not in wanted:
                continue
            if member.size > 8 * 1024 * 1024:
                raise ValueError(f"Unexpectedly large font asset: {member.name}")
            if path.name in selected:
                raise ValueError(f"Duplicate font asset: {path.name}")
            stream = members.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable font asset: {member.name}")
            selected[path.name] = stream.read()
    missing = wanted - selected.keys()
    if missing:
        raise ValueError(f"Font archive is missing: {', '.join(sorted(missing))}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="font-assets-", dir=output.parent) as tmp:
        staged = Path(tmp)
        for name, data in selected.items():
            (staged / name).write_bytes(data)
        # Validate actual font data before publishing any files.
        from PIL import ImageFont
        for name in REGULAR_FONTS:
            ImageFont.truetype(str(staged / name), 22)
        output.mkdir(parents=True, exist_ok=True)
        for name in sorted(selected):
            (staged / name).replace(output / name)
            (output / name).chmod(0o644)
    result = {
        "archive": str(Path(archive).resolve()),
        "archive_sha256": file_digest(archive),
        "fonts": {name: file_digest(output / name) for name in REGULAR_FONTS},
        "license": "LICENSE",
    }
    (output / "asset_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def git_blob_sha1(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_corpus(path: str | Path) -> dict:
    path = Path(path)
    actual = file_digest(path, "md5")
    result = {"path": str(path.resolve()), "bytes": path.stat().st_size,
              "md5": actual, "expected_md5": CORPUS_MD5}
    if actual != CORPUS_MD5 or path.stat().st_size != CORPUS_BYTES:
        raise ValueError("Corpus does not match the official SLR11 archive: "
                         + json.dumps(result))
    return result


def check_eng(path: str | Path) -> dict:
    path = Path(path)
    actual = git_blob_sha1(path)
    result = {"path": str(path.resolve()), "bytes": path.stat().st_size,
              "git_blob_sha1": actual, "sha256": file_digest(path)}
    if actual != ENG_GIT_BLOB_SHA1 or path.stat().st_size != ENG_BYTES:
        raise ValueError("eng.traineddata does not match the pinned tessdata_best blob: "
                         + json.dumps(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fonts = sub.add_parser("fonts", help="Prepare three Regular fonts and LICENSE")
    fonts.add_argument("--archive", required=True, type=Path)
    fonts.add_argument("--output", required=True, type=Path)
    check = sub.add_parser("check", help="Check local official corpus and/or English model")
    check.add_argument("--corpus", type=Path)
    check.add_argument("--eng", type=Path)
    args = parser.parse_args()
    if args.command == "fonts":
        result = prepare_fonts(args.archive, args.output)
    else:
        if args.corpus is None and args.eng is None:
            parser.error("check requires --corpus and/or --eng")
        result = {}
        if args.corpus is not None:
            result["corpus"] = check_corpus(args.corpus)
        if args.eng is not None:
            result["eng"] = check_eng(args.eng)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
