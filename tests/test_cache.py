import asyncio
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from spider_trace.cache import SQLCacheSession
from spider_trace.sql import execute_query


class Found:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MemoryClient:
    def __init__(self):
        self.values = {}

    async def find(self, task, path):
        return Found(self.values.get((task, tuple(path))))

    async def insert(self, task, path, value, sandbox, runtime):
        assert sandbox is None
        self.values[task, tuple(path)] = value
        return SimpleNamespace(value_inserted=True)

    async def close(self):
        pass


class CacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "test.sqlite"
        with sqlite3.connect(self.path) as conn:
            conn.execute("CREATE TABLE items (id INTEGER)")
            conn.execute("INSERT INTO items VALUES (1)")
        self.client = MemoryClient()
        self.session = SQLCacheSession(self.path, client=self.client)
        self.executions = 0

    def query(self, path, params):
        self.executions += 1
        return execute_query(path, params.get("sql"))

    async def test_miss_hit_equal_and_exact_arguments(self):
        parameters = {"sql": "SELECT * FROM items"}
        first, miss = await self.session.execute(parameters, self.query)
        second, hit = await self.session.execute(parameters, self.query)
        self.assertEqual(first, second)
        self.assertEqual((miss["status"], hit["status"]), ("miss", "hit"))
        self.assertEqual(self.executions, 1)
        await self.session.execute({"sql": "SELECT * FROM items;"}, self.query)
        self.assertEqual(self.executions, 2)

    async def test_errors_are_not_cached(self):
        for _ in range(2):
            result, metadata = await self.session.execute({"sql": "SELECT * FROM missing"}, self.query)
            self.assertFalse(result["ok"])
            self.assertFalse(metadata["inserted"])
        self.assertEqual(self.executions, 2)

    async def test_database_and_namespace_isolation(self):
        other = SQLCacheSession(self.path, client=self.client, namespace="another")
        self.assertNotEqual(self.session.task_id, other.task_id)
        with sqlite3.connect(self.path) as conn:
            conn.execute("INSERT INTO items VALUES (2)")
        with self.assertRaisesRegex(ValueError, "changed"):
            await self.session.execute({"sql": "SELECT 1"}, self.query)
        newer = SQLCacheSession(self.path, client=self.client)
        self.assertNotEqual(self.session.task_id, newer.task_id)

    async def test_same_session_serializes_requests(self):
        results = await asyncio.gather(*[
            self.session.execute({"sql": "SELECT 1"}, self.query) for _ in range(3)])
        self.assertEqual([metadata["status"] for _, metadata in results], ["miss", "hit", "hit"])
        self.assertEqual(self.executions, 1)

    async def test_bad_cached_identity_fails(self):
        await self.session.execute({"sql": "SELECT 1"}, self.query)
        for key in self.client.values:
            self.client.values[key] = json.dumps({"key": "wrong"})
        with self.assertRaisesRegex(ValueError, "identity"):
            await self.session.execute({"sql": "SELECT 1"}, self.query)

    async def test_sidecar_rejected(self):
        Path(str(self.path) + "-wal").touch()
        with self.assertRaisesRegex(ValueError, "immutable"):
            SQLCacheSession(self.path, client=self.client)

    async def test_disabled_and_invalid_modes(self):
        with patch.dict("os.environ", {"SPIDER_CACHE_MODE": "off"}):
            self.assertIsNone(SQLCacheSession.from_environment(self.path))
        with patch.dict("os.environ", {"SPIDER_CACHE_MODE": "typo"}):
            with self.assertRaises(ValueError):
                SQLCacheSession.from_environment(self.path)

    async def test_cache_failure_does_not_silently_execute(self):
        async def broken(*args):
            raise ConnectionError("server unavailable")
        self.client.find = broken
        with self.assertRaises(ConnectionError):
            await self.session.execute({"sql": "SELECT 1"}, self.query)
        self.assertEqual(self.executions, 0)
