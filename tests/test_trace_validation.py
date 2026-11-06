import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from spider_trace.trace_validation import assess_trajectory, validate_trajectory


class TraceValidationTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            {"event": "start", "timestamp_ns": 1},
            {"event": "assistant", "timestamp_ns": 2, "text": "<sql>SELECT 1;</sql>"},
            {"event": "tool", "timestamp_ns": 3, "tool_name": "execute_sql",
             "db_id": "test", "arguments": {"sql": "SELECT 1;"}, "execution_ms": 1.0,
             "sql": "SELECT 1;", "result": {"ok": True, "truncated": False,
                                             "columns": ["1"], "rows": [[1]]}},
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
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("PRESERVED", result.stdout)
            self.assertEqual(len(path.read_text().splitlines()), len(records + bad))
            result = subprocess.run(command + ["--strict"], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing final solution", result.stdout)
            self.assertIn("1/2 invalid", result.stderr)
            path.write_text("".join(json.dumps(e) + "\n" for e in bad[:-1]))
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("INVALID TRACE", result.stdout)
            self.assertIn("0/0 evaluated; 1 not evaluated", result.stdout)

    def test_unsuccessful_models_have_intact_traces(self):
        self.events[3]["text"] = "<|im_end|>"
        self.events[2]["result"]["ok"] = False
        self.events[2]["result"]["error"] = "no such table"
        self.assertEqual(validate_trajectory(self.events, require_verified_solution=False), [])
        without_tools = [e for e in self.events if e["event"] != "tool"]
        self.assertEqual(validate_trajectory(without_tools, require_verified_solution=False), [])
        self.assertTrue(validate_trajectory(without_tools))

    def test_tool_payload_integrity(self):
        for field in ("tool_name", "db_id", "arguments", "sql", "execution_ms", "result"):
            with self.subTest(field=field):
                events = copy.deepcopy(self.events)
                del events[2][field]
                report = assess_trajectory(events)
                self.assertTrue(report.integrity_errors)
                self.assertIsNone(report.protocol_errors)
        for elapsed in (-1, float("nan"), float("inf"), True, "1"):
            events = copy.deepcopy(self.events)
            events[2]["execution_ms"] = elapsed
            self.assertTrue(assess_trajectory(events).integrity_errors)
        for result in ({"ok": True}, {"ok": False}, {"ok": "true"},
                       {"ok": True, "columns": ["x"], "rows": [[1, 2]], "truncated": False}):
            events = copy.deepcopy(self.events)
            events[2]["result"] = result
            self.assertTrue(assess_trajectory(events).integrity_errors)
        self.events[2]["arguments"]["sql"] = "SELECT 2;"
        self.assertTrue(assess_trajectory(self.events).integrity_errors)

    def test_invalid_model_arguments_are_preserved(self):
        self.events[2].update(arguments={}, sql=None, result={"ok": False, "error": "SQL must be a string"})
        report = assess_trajectory(self.events)
        self.assertEqual(report.integrity_errors, [])
        self.assertTrue(report.protocol_errors)

    def test_assessment_does_not_modify_events(self):
        original = copy.deepcopy(self.events)
        assess_trajectory(self.events)
        self.assertEqual(self.events, original)
