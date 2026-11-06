import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from spider_trace.trace_validation import validate_trajectory


class TraceValidationTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            {"event": "start", "timestamp_ns": 1},
            {"event": "assistant", "timestamp_ns": 2, "text": "<sql>SELECT 1;</sql>"},
            {"event": "tool", "timestamp_ns": 3, "tool_name": "execute_sql",
             "sql": "SELECT 1;", "result": {"ok": True, "truncated": False}},
            {"event": "assistant", "timestamp_ns": 4, "text": "<solution>SELECT 1;</solution>"},
            {"event": "complete", "timestamp_ns": 5, "response_ids": [1, 2], "response_mask": [1, 0]}]

    def test_valid_and_unsorted(self):
        self.assertEqual(validate_trajectory(self.events), [])
        self.assertEqual(validate_trajectory(list(reversed(self.events))), [])

    def test_empty_answer_regression(self):
        self.events[3]["text"] = "<|im_end|>"
        self.assertIn("missing final solution", ";".join(validate_trajectory(self.events)))

    def test_failed_truncated_or_unverified_query(self):
        for changes in ({"ok": False}, {"truncated": True}):
            events = copy.deepcopy(self.events)
            events[2]["result"].update(changes)
            self.assertTrue(validate_trajectory(events))
        self.events[3]["text"] = "<solution>SELECT 2;</solution>"
        self.assertTrue(validate_trajectory(self.events))

    def test_no_credit_for_solution_inside_thinking(self):
        self.events[3]["text"] = "<think><solution>SELECT 1;</solution></think>"
        self.assertTrue(validate_trajectory(self.events))

    def test_missing_complete_error_and_bad_mask(self):
        self.assertTrue(validate_trajectory([]))
        self.assertTrue(validate_trajectory(self.events[:-1]))
        self.assertTrue(validate_trajectory(self.events + [{"event": "error", "timestamp_ns": 6}]))
        self.events[-1]["response_mask"] = [1]
        self.assertTrue(validate_trajectory(self.events))

    def test_cli_valid_and_mixed_run(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(root))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events-test.jsonl"
            records = [dict(event, trajectory_id="good") for event in self.events]
            path.write_text("".join(json.dumps(e) + "\n" for e in records))
            command = [sys.executable, str(root / "scripts/inspect_trace.py"), directory]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            bad = [dict(event, trajectory_id="bad") for event in self.events]
            bad[3]["text"] = "<|im_end|>"
            path.write_text("".join(json.dumps(e) + "\n" for e in records + bad))
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing final solution", result.stdout)
            self.assertIn("1/2 invalid", result.stderr)
