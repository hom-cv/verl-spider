import sqlite3
import tempfile
import unittest
from pathlib import Path

from spider_trace.exploration import SYSTEM, exploration_prompt, tool_bindings
from spider_trace.sql import list_tables, get_table_schema


class ExplorationTests(unittest.TestCase):
    def test_removes_legacy_schema_without_mutating_input(self):
        messages = [{"role": "system", "content": "Use supplied schema. Five calls."},
                    {"role": "user", "content": "Schema:\nCREATE TABLE secret(id);\n\nQuestion: Count rows"}]
        upgraded = exploration_prompt(messages)
        self.assertEqual(upgraded[0]["content"], SYSTEM)
        self.assertEqual(upgraded[1]["content"], "Question: Count rows")
        self.assertNotIn("secret", str(upgraded))
        self.assertIn("CREATE TABLE secret", messages[1]["content"])
        self.assertEqual(exploration_prompt(upgraded), upgraded)

    def test_fails_closed_if_old_schema_cannot_be_removed(self):
        with self.assertRaises(ValueError):
            exploration_prompt([{"role": "user", "content": "Schema:\nsecret"}])

    def test_all_tools_bound_to_same_database(self):
        bindings = tool_bindings("tiny")
        self.assertEqual(set(bindings), {"list_tables", "get_table_schema", "execute_sql"})
        self.assertTrue(all(b["create_kwargs"]["db_id"] == "tiny" for b in bindings.values()))

    def test_discovery_is_repeatable_and_parameterized(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "test.sqlite"
            with sqlite3.connect(db) as conn:
                conn.executescript('CREATE TABLE z(id INTEGER PRIMARY KEY); '
                                   'CREATE TABLE a(z_id INTEGER REFERENCES z(id));')
            self.assertEqual(list_tables(db), {"ok": True, "tables": ["a", "z"]})
            schema = get_table_schema(db, ["z", "a"])
            self.assertEqual(schema, get_table_schema(db, ["a", "z", "a"]))
            self.assertIn("REFERENCES z", schema["tables"][0]["create_sql"])
            for names in [[], "a", [None], ["missing"], ["a' OR 1=1 --"], ["a"] * 11]:
                self.assertFalse(get_table_schema(db, names)["ok"])
            self.assertEqual(list_tables(db)["tables"], ["a", "z"])
