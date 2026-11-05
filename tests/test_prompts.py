import unittest

from spider_trace.prompts import TOOL_PROTOCOL, with_tool_protocol


class PromptTests(unittest.TestCase):
    def test_upgrades_existing_data_without_mutation(self):
        original = [{"role": "system", "content": "SQL instructions"},
                    {"role": "user", "content": "Count singers"}]
        upgraded = with_tool_protocol(original)
        self.assertIn(TOOL_PROTOCOL, upgraded[0]["content"])
        self.assertEqual(original[0]["content"], "SQL instructions")
        self.assertEqual(upgraded[1], original[1])

    def test_idempotent_for_newly_prepared_data(self):
        messages = [{"role": "system", "content": TOOL_PROTOCOL}]
        self.assertEqual(with_tool_protocol(messages), messages)

    def test_adds_missing_system_message(self):
        messages = [{"role": "user", "content": "Count singers"}]
        upgraded = with_tool_protocol(messages)
        self.assertEqual(upgraded[0], {"role": "system", "content": TOOL_PROTOCOL})
        self.assertEqual(upgraded[1:], messages)
