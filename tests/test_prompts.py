import sqlite3
import tempfile
import unittest
from pathlib import Path

from spider_trace.prompts import build_messages, question_from_messages, tool_bindings


class PromptTests(unittest.TestCase):
    def test_single_prompt_contains_real_schema_and_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "tiny.sqlite"
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE singers(id INTEGER PRIMARY KEY)")
            messages = build_messages("Count singers", db)
            text = messages[0]["content"]
            for expected in ("Task Overview:", "CREATE TABLE singers", "<sql>",
                             "<observation>", "<solution>", "50 rows"):
                self.assertIn(expected, text)
            self.assertNotIn("<tool_call>", text)
            self.assertNotIn("{db_details}", text)
            self.assertEqual(build_messages(question_from_messages(messages), db), messages)

    def test_legacy_question_extraction(self):
        messages = [{"role": "user", "content": "Schema:\nold schema\n\nQuestion: Count singers"}]
        self.assertEqual(question_from_messages(messages), "Count singers")
        self.assertIn("old schema", messages[0]["content"])
        with self.assertRaises(ValueError):
            question_from_messages([{"role": "user", "content": "Schema:\nbroken"}])

    def test_only_sql_tool_is_bound(self):
        self.assertEqual(tool_bindings("tiny"), {
            "execute_sql": {"create_kwargs": {"db_id": "tiny"}}})
