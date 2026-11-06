import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

from spider_trace.provenance import record_run, sha256, verify_database


class ProvenanceTests(unittest.TestCase):
    def test_hash_validation_and_dirty_source_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "databases/tiny/tiny.sqlite"
            db.parent.mkdir(parents=True)
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE t(id INTEGER)")
            sources = {"database_root": str(root / "databases"), "db_id": "tiny",
                       "database_sha256": sha256(db)}
            self.assertEqual(verify_database(sources), sha256(db))
            with self.assertRaises(ValueError):
                verify_database(dict(sources, database_sha256="wrong"))
            wal = Path(str(db) + "-wal")
            wal.touch()
            with self.assertRaises(ValueError):
                verify_database(sources)
            wal.unlink()
            (root / "data").mkdir()
            (root / "data/sources.json").write_text(json.dumps(sources))
            (root / "data/mini.parquet").write_bytes(b"test dataset")
            (root / "spider_trace").mkdir()
            source = root / "spider_trace/example.py"
            source.write_text("original = True\n")
            for name in ("pyproject.toml", "requirements-gpu.txt"):
                (root / name).touch()
            def git(*args):
                subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
            git("init")
            git("add", "spider_trace/example.py")
            git("-c", "user.name=Test", "-c", "user.email=test@example.com",
                "-c", "commit.gpgsign=false", "commit", "-m", "fixture")
            source.write_text("original = False\n")
            (root / "spider_trace/new.py").write_text("new = True\n")
            output = root / "run"
            record_run(root, output)
            self.assertEqual((output / "source/spider_trace/example.py").read_text(), source.read_text())
            self.assertTrue((output / "source/spider_trace/new.py").exists())
            self.assertIn("original = False", (output / "repo-diff.patch").read_text())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["source_sha256"]["spider_trace/example.py"], sha256(source))
