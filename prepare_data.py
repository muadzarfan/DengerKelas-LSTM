"""Download and extract the FLEURS recordings used by the LSTM experiment."""

import argparse
import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
AUDIO_ROOT = ROOT / "data/raw/fleurs"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def check_archive(path, spec):
    if path.stat().st_size != spec["size"] or digest(path) != spec["sha256"]:
        raise ValueError(f"Archive checksum or size mismatch: {path}")


def download(path, spec):
    temporary = path.with_suffix(".part")
    print(f"Downloading {path.name} ({spec['size'] / 1e6:.0f} MB)", flush=True)
    try:
        with urlopen(spec["url"], timeout=120) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        check_archive(temporary, spec)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def missing_audio(split, archive_split):
    manifest = ROOT / f"data/manifests/fleurs_{split}.jsonl"
    pending = {}
    for row in read_jsonl(manifest):
        name = row["archive_member"]
        if not re.fullmatch(rf"{archive_split}/[0-9]+\.wav", name):
            raise ValueError(f"Unexpected archive member: {name}")
        path = AUDIO_ROOT / "audio" / name
        checksum = row["audio_sha256"]
        if not path.is_file() or digest(path) != checksum:
            pending[name] = (path, checksum)
    return pending


def extract_audio(archive, pending):
    with tarfile.open(archive, mode="r|gz") as source:
        for member in source:
            if member.name not in pending:
                continue
            path, checksum = pending[member.name]
            if not member.isfile():
                raise ValueError(f"Invalid audio member: {member.name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".part")
            try:
                with source.extractfile(member) as audio, temporary.open("wb") as handle:
                    shutil.copyfileobj(audio, handle, length=1024 * 1024)
                if digest(temporary) != checksum:
                    raise ValueError(f"Audio checksum mismatch: {member.name}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            del pending[member.name]
            if not pending:
                break
    if pending:
        raise ValueError(f"{archive.name} is missing {len(pending)} required recordings")


def prepare(archive_dir=None):
    specs = json.loads((ROOT / "data/sources.json").read_text(encoding="utf-8"))
    AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    for split, archive_split in (("train", "train"), ("validation", "dev")):
        pending = missing_audio(split, archive_split)
        if not pending:
            print(f"{split}: audio already verified", flush=True)
            continue
        name = f"{archive_split}.tar.gz"
        archive = (archive_dir or AUDIO_ROOT) / name
        if archive_dir is None and not archive.exists():
            download(archive, specs[name])
        else:
            check_archive(archive, specs[name])
        print(f"{split}: extracting {len(pending)} recordings", flush=True)
        extract_audio(archive, pending)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, help="Use existing train.tar.gz and dev.tar.gz archives.")
    prepare(parser.parse_args().archive_dir)
