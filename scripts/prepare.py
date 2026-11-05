"""Download pinned snapshots and prepare a small Spider validation subset."""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, snapshot_download

from spider_trace.prompts import TOOL_PROTOCOL
from spider_trace.sql import connect, execute_query, find_validation_database_root

SYSTEM = """You are a SQL assistant. Solve the question using the supplied SQLite schema.
Explain your plan briefly, then call execute_sql to test your query before submitting.
Inspect the returned rows or errors; revise and call the tool again when necessary.
Use one read-only query per call, deterministic expressions, and explicit ORDER BY
with unique tie breakers when selecting a limited number of rows. Do not use random
or time functions. End with your answer query in <final_sql>...</final_sql>.
Do not invent tool results. You may make at most five tool calls."""
SYSTEM += "\n\n" + TOOL_PROTOCOL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--db-id", default="concert_singer")
    parser.add_argument("--database-root", type=Path, help="Existing Spider database/ directory; skips archive download")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    data.mkdir(exist_ok=True)
    lockfile = data / "sources.json"
    api = HfApi()
    if lockfile.exists():
        sources = json.loads(lockfile.read_text())
        if sources["model_id"] != args.model:
            parser.error("Model differs from data/sources.json; use a separate checkout for another model")
    else:
        sources = {"model_id": args.model, "model_revision": api.model_info(args.model).sha,
                   "dataset_revision": api.dataset_info("xlangai/spider").sha,
                   "database_revision": api.dataset_info("HAL-9001/spider-databases").sha}
        lockfile.write_text(json.dumps(sources, indent=2) + "\n")
    snapshot_download(args.model, revision=sources["model_revision"], local_dir=root / "models" / "qwen",
                      ignore_patterns=["*.gguf", "*.bin", "*.h5", "*.msgpack"])
    if args.database_root:
        dbroot = args.database_root.resolve()
    else:
        archive_dir = Path(snapshot_download("HAL-9001/spider-databases", repo_type="dataset",
                         revision=sources["database_revision"], allow_patterns=["spider_data.zip"]))
        archive = archive_dir / "spider_data.zip"
        sources["archive_sha256"] = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(archive.read_bytes()).hexdigest()
        destination = (data / "spider-files").resolve()
        destination.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                target = (destination / item.filename).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError("Unsafe archive path")
            z.extractall(destination)
        dbroot = find_validation_database_root(destination, args.db_id)
    dbpath = dbroot / args.db_id / f"{args.db_id}.sqlite"
    with connect(dbpath) as conn:
        schema = "\n".join(row[0] for row in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"))
    raw = load_dataset("xlangai/spider", revision=sources["dataset_revision"], split="validation")
    rows = []
    for index, item in enumerate(raw):
        if item["db_id"] != args.db_id:
            continue
        gold = execute_query(dbpath, item["query"], max_rows=10000, max_chars=1000000)
        if not gold["ok"] or gold["truncated"]:
            continue
        rows.append({"data_source": "spider_smoke", "agent_name": "spider_agent", "ability": "sql",
                     "prompt": [{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {item['question']}"}],
                     "reward_model": {"style": "rule", "ground_truth": json.dumps({
                         "db_id": args.db_id, "sql": item["query"],
                         "ordered": bool(re.search(r"\border\s+by\b", item["query"], re.I))})},
                     "extra_info": {"index": index, "db_id": args.db_id, "need_tools_kwargs": True,
                                    "tools_kwargs": {"execute_sql": {"create_kwargs": {"db_id": args.db_id}}}}})
        if len(rows) >= args.limit:
            break
    if not rows:
        raise ValueError("No usable questions found")
    Dataset.from_list(rows).to_parquet(str(data / "mini.parquet"))
    sources.update(database_root=str(dbroot), samples=len(rows), db_id=args.db_id,
                   database_sha256=hashlib.sha256(dbpath.read_bytes()).hexdigest())
    lockfile.write_text(json.dumps(sources, indent=2) + "\n")
    print(f"Prepared {len(rows)} questions. Submit scripts/perlmutter.sbatch with your account.")


if __name__ == "__main__":
    main()
