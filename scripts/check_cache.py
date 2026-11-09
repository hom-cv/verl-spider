"""CPU-only integration check against a running TVCache server and real SQLite."""
import argparse
import asyncio
import json
from pathlib import Path
import sqlite3
import tempfile
import time
from uuid import uuid4

from tvclient import TVCacheClient
from spider_trace.cache import SQLCacheSession
from spider_trace.sql import execute_query


async def check(host, port, trace_root=Path("runs/cache-tests")):
    run_id = uuid4().hex
    trace_dir = trace_root / run_id
    trace_dir.mkdir(parents=True, exist_ok=False)
    trace_path = trace_dir / "cache-events.jsonl"
    print(f"Trace: {trace_path.resolve()}", flush=True)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "probe.sqlite"
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE probe (value INTEGER)")
            conn.execute("INSERT INTO probe VALUES (42)")
        namespace = "check-" + run_id
        executions = 0

        def query(path, parameters):
            nonlocal executions
            executions += 1
            return execute_query(path, parameters["sql"])

        results, statuses = [], []
        # Separate connections simulate different rollouts sharing server values.
        for call_index in range(2):
            async with TVCacheClient(host=host, port=port, retries=0) as client:
                session = SQLCacheSession(path, client=client, namespace=namespace)
                parameters = {"sql": "SELECT value FROM probe"}
                before = executions
                start = time.perf_counter()
                try:
                    result, metadata = await session.execute(parameters, query)
                except Exception as exc:
                    with trace_path.open("a") as stream:
                        stream.write(json.dumps({"event": "cache_test_error", "call_index": call_index,
                                                 "error": str(exc), "timestamp_ns": time.time_ns()}) + "\n")
                    raise
                elapsed_ms = (time.perf_counter() - start) * 1000
                record = {"event": "cache_test_call", "synthetic": True,
                          "call_index": call_index, "timestamp_ns": time.time_ns(),
                          "namespace": namespace, "server": {"host": host, "port": port},
                          "database_sha256": session.db_hash, "arguments": parameters,
                          "cache": metadata, "execution_ms": elapsed_ms,
                          "sql_executed": executions > before, "result": result}
                with trace_path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(f"Call {call_index + 1}: {metadata['status'].upper()} | "
                      f"SQL executed={record['sql_executed']} | "
                      f"lookup={metadata['lookup_ms']:.3f} ms | total={elapsed_ms:.3f} ms", flush=True)
                results.append(result)
                statuses.append(metadata["status"])
        assert statuses == ["miss", "hit"], statuses
        assert executions == 1, executions
        assert results[0] == results[1] and results[0]["rows"] == [[42]], results
        print("PASS: real TVCache + TVClient + SQLite; miss then cross-connection hit, identical results, one SQL execution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7878)
    parser.add_argument("--trace-root", type=Path, default=Path("runs/cache-tests"))
    args = parser.parse_args()
    asyncio.run(check(args.host, args.port, args.trace_root))
