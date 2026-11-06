"""veRL adapter for the explicit SQL action tags in the paper-style prompt."""
import json

from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser

from spider_trace.sql_protocol import extract_sql_call


@ToolParser.register("sql")
class SQLBlockParser(ToolParser):
    async def extract_tool_calls(self, responses_ids):
        text = self.tokenizer.decode(responses_ids, skip_special_tokens=False)
        sql = extract_sql_call(text)
        calls = [] if sql is None else [FunctionCall(
            name="execute_sql", arguments=json.dumps({"sql": sql}))]
        return text, calls
