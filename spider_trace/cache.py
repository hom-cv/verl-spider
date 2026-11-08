"""Opt-in TVClient value caching for immutable, read-only SQLite databases.

One session per tool instance/rollout; no sandbox snapshots or stateful history.
Cache failures are fatal rather than silently biasing benchmark measurements.
"""
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time

from spider_trace import sql as sql_module
from spider_trace.limits import TOOL_MAX_ROWS, TOOL_RESULT_CHARS


def digest(value):
    return hashlib.sha256(value).hexdigest()


class SQLCacheSession:
    def __init__(self, path, *, client=None, namespace="spider-sql-v1"):
        self.path = Path(path)
        self.client = client
        self.lock = asyncio.Lock()
        self.signature = self._signature()
        self.db_hash = digest(self.path.read_bytes())
        if self._signature() != self.signature:
            raise ValueError("Database changed while hashing")
        identity = {"namespace": namespace, "database": self.db_hash,
                    "sqlite": sqlite3.sqlite_version,
                    "implementation": digest(Path(sql_module.__file__).read_bytes()),
                    "max_rows": TOOL_MAX_ROWS, "max_chars": TOOL_RESULT_CHARS}
        self.task_id = "spider-sql:" + digest(json.dumps(identity, sort_keys=True).encode())

    def _signature(self):
        if any(Path(str(self.path) + suffix).exists() for suffix in ("-wal", "-journal")):
            raise ValueError("Caching requires an immutable database without WAL/journal sidecars")
        stat = self.path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    @classmethod
    def from_environment(cls, path):
        mode = os.environ.get("SPIDER_CACHE_MODE", "off")
        if mode == "off":
            return None
        if mode != "tvcache":
            raise ValueError("SPIDER_CACHE_MODE must be off or tvcache")
        from tvclient import TVCacheClient
        client = TVCacheClient(host=os.environ.get("SPIDER_CACHE_HOST", "127.0.0.1"),
                               port=int(os.environ.get("SPIDER_CACHE_PORT", "7878")),
                               retries=0)
        return cls(path, client=client,
                   namespace=os.environ.get("SPIDER_CACHE_NAMESPACE", "spider-sql-v1"))

    async def execute(self, parameters, run_query):
        # TVClient requests on a connection must be sequential.
        async with self.lock:
            if self._signature() != self.signature:
                raise ValueError("Database changed during cached rollout")
            key = digest(json.dumps(parameters, sort_keys=True, ensure_ascii=True).encode())
            lookup_start = time.perf_counter()
            async with await self.client.find(self.task_id, [key]) as found:
                lookup_ms = (time.perf_counter() - lookup_start) * 1000
                if found.value is not None:
                    entry = json.loads(found.value)
                    if entry.get("key") != key or entry.get("task_id") != self.task_id:
                        raise ValueError("Cache value identity mismatch")
                    result = entry["result"]
                    if not isinstance(result, dict) or result.get("ok") is not True:
                        raise ValueError("Invalid cached SQL result")
                    return result, {"status": "hit", "key": key, "task_id": self.task_id,
                                    "lookup_ms": lookup_ms, "sql_execution_ms": 0.0}
            start = time.perf_counter()
            result = await asyncio.to_thread(run_query, self.path, parameters)
            sql_ms = (time.perf_counter() - start) * 1000
            if self._signature() != self.signature:
                raise ValueError("Database changed during SQL execution")
            inserted = False
            # Errors may be transient; do not cache them. Successful bounded
            # results include their truncation flag, so replay preserves output.
            if result.get("ok") is True:
                value = json.dumps({"key": key, "task_id": self.task_id, "result": result})
                response = await self.client.insert(self.task_id, [key], value, None,
                                                    math.ceil(sql_ms))
                inserted = response.value_inserted
            return result, {"status": "miss", "key": key, "task_id": self.task_id,
                            "lookup_ms": lookup_ms, "sql_execution_ms": sql_ms,
                            "inserted": inserted}

    async def close(self):
        await self.client.close()
