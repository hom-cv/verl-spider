"""Verify prepared inputs and capture the code actually used by a run."""
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from spider_trace.sql import database_path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_database(sources):
    path = database_path(sources["database_root"], sources["db_id"])
    expected = sources.get("database_sha256")
    actual = sha256(path)
    if not expected or actual != expected:
        raise ValueError("Database hash differs from data/sources.json; verify/reprepare the input before running")
    if Path(str(path) + "-wal").exists():
        raise ValueError("Database has a WAL sidecar; use a fixed, checkpointed database snapshot")
    return actual


def record_run(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    sources = json.loads((root / "data/sources.json").read_text())
    database_hash = verify_database(sources)
    destination.mkdir(parents=True, exist_ok=True)

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)

    (destination / "repo-revision.txt").write_bytes(git("rev-parse", "HEAD"))
    (destination / "repo-status.txt").write_bytes(git("status", "--porcelain"))
    (destination / "repo-diff.patch").write_bytes(git("diff", "--binary", "HEAD"))
    # Include uncommitted/new runtime source too. Never snapshot caches, models,
    # datasets, credentials, or arbitrary untracked files outside these paths.
    paths = set()
    for directory in ("spider_trace", "scripts", "config"):
        paths.update(p for p in (root / directory).rglob("*")
                     if p.suffix in (".py", ".sh", ".yaml", ".sbatch"))
    paths.update(root / name for name in ("pyproject.toml", "requirements-gpu.txt"))
    hashes = {}
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected a regular source file: {path}")
        relative = path.relative_to(root)
        target = destination / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        target.write_bytes(data)
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
    manifest = {"database_sha256": database_hash,
                "dataset_sha256": sha256(root / "data/mini.parquet"),
                "sqlite_version": sqlite3.sqlite_version, "source_sha256": hashes}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    record_run(Path.cwd(), sys.argv[1])
