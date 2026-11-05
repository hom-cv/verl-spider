import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from spider_trace.sql import database_path, execute_query
from spider_trace.reward import compute_score


class SQLTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "tiny").mkdir()
        self.db = self.root / "tiny" / "tiny.sqlite"
        with sqlite3.connect(self.db) as conn:
            conn.executescript("CREATE TABLE t(id INTEGER, name TEXT); INSERT INTO t VALUES (1,'Ada'),(2,'Bo'),(3,'Bo');")

    def tearDown(self):
        self.temp.cleanup()

    def test_repeatable_query_and_cte(self):
        sql = "WITH x AS (SELECT * FROM t) SELECT name FROM x ORDER BY id"
        a = execute_query(self.db, sql)
        self.assertEqual(a, execute_query(self.db, sql))
        self.assertEqual(a["rows"], [["Ada"], ["Bo"], ["Bo"]])

    def test_writes_and_nondeterminism_denied(self):
        for sql in ["DELETE FROM t", "DROP TABLE t", "PRAGMA user_version=1",
                    "ATTACH DATABASE ':memory:' AS other", "SELECT random()", "SELECT CURRENT_TIMESTAMP",
                    "SELECT datetime('now')", "SELECT group_concat(name) FROM t",
                    "SELECT 1; DELETE FROM t"]:
            with self.subTest(sql=sql):
                self.assertFalse(execute_query(self.db, sql)["ok"])
        self.assertEqual(execute_query(self.db, "SELECT count(*) FROM t")["rows"], [[3]])

    def test_truncation_and_instruction_budget(self):
        result = execute_query(self.db, "SELECT * FROM t ORDER BY id", max_rows=1)
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["rows"]), 1)
        result = execute_query(self.db, "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT sum(n) FROM x")
        self.assertFalse(result["ok"])

    def test_path_traversal(self):
        with self.assertRaises(ValueError):
            database_path(self.root, "../tiny")

    def test_reward_duplicates_order_and_truncation(self):
        with patch.dict(os.environ, SPIDER_DB_ROOT=str(self.root)):
            gold = json.dumps({"db_id": "tiny", "sql": "SELECT name FROM t", "ordered": False})
            self.assertEqual(compute_score("", "<final_sql>SELECT name FROM t ORDER BY id DESC</final_sql>", gold), 1)
            self.assertEqual(compute_score("", "<final_sql>SELECT DISTINCT name FROM t</final_sql>", gold), 0)
            self.assertEqual(compute_score("", "no final answer", gold), 0)
            gold = json.dumps({"db_id": "tiny", "sql": "SELECT name FROM t ORDER BY id", "ordered": True})
            self.assertEqual(compute_score("", "<final_sql>SELECT name FROM t ORDER BY id DESC</final_sql>", gold), 0)


if __name__ == "__main__":
    unittest.main()
