"""Check the existing parser and SQL tool without loading model weights."""
import asyncio
import json
import os
import tempfile
from pathlib import Path

from transformers import AutoTokenizer
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.utils.tool_registry import initialize_tools_from_config

from spider_trace.events import trajectory
from spider_trace.prompts import build_messages
from spider_trace.sql import database_path
from spider_trace.sql_parser import SQLBlockParser  # Registers the SQL-tag parser.


async def main():
    root = Path(__file__).resolve().parents[1]
    sources = json.loads((root / "data/sources.json").read_text())
    os.environ["SPIDER_DB_ROOT"] = sources["database_root"]
    # Synthetic calls must not be counted as model-generated cache workload.
    diagnostic_root = root / "runs" / "diagnostics"
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="tool-check-", dir=diagnostic_root))
    os.environ["TRACE_DIR"] = str(output)
    print(f"Diagnostic output (synthetic, not a rollout): {output}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(root / "models/qwen", local_files_only=True)
    messages = build_messages("How many singers do we have?", database_path(
        sources["database_root"], sources["db_id"]))
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if "<tool_call>" in rendered or "<sql>" not in rendered:
        raise RuntimeError("Rendered prompt does not match the SQL-tag protocol")
    print("PASS: rendered prompt uses SQL tags without JSON tool instructions", flush=True)
    parser = ToolParser.get_tool_parser("sql", tokenizer)
    tools = {tool.name: tool for tool in
             initialize_tools_from_config(str(root / "config/tools.yaml"))}
    if "execute_sql" not in tools:
        raise RuntimeError(f"Tool registration failed: found {list(tools)}")
    print("PASS: execute_sql is registered", flush=True)

    sql = "SELECT 1 AS probe;"
    text = '<think>Check the connection.</think><sql>' + sql + '</sql>'
    _, calls = await parser.extract_tool_calls(tokenizer.encode(text, add_special_tokens=False))
    if len(calls) != 1 or calls[0].name != "execute_sql":
        raise RuntimeError(f"Parser failed to recover the known-valid call: {calls}")
    arguments = json.loads(calls[0].arguments)
    if arguments != {"sql": sql}:
        raise RuntimeError(f"Parser changed the SQL arguments: {arguments}")
    print("PASS: parser recovered execute_sql and the exact SQL", flush=True)

    tool = tools[calls[0].name]
    token = trajectory.set("synthetic-tool-check")
    instance_id = None
    try:
        instance_id, _ = await tool.create(create_kwargs={"db_id": sources["db_id"]})
        response, _, _ = await tool.execute(instance_id, arguments)
        print("SQL result:", response.text, flush=True)
        if response.text != "<observation>\n| probe |\n| --- |\n| 1 |\n</observation>":
            raise RuntimeError("SQL tool did not return the expected probe result")
    finally:
        if instance_id is not None:
            await tool.release(instance_id)
        trajectory.reset(token)
    print("PASS: SQL tool executed the parsed call and returned [[1]]", flush=True)
    _, final_calls = await parser.extract_tool_calls(tokenizer.encode(
        '<solution>SELECT 1 AS probe;</solution>', add_special_tokens=False))
    if final_calls:
        raise RuntimeError("Final solution must not execute a tool")
    print("Wiring check passed. This does not test whether the model chooses to call tools.")


if __name__ == "__main__":
    asyncio.run(main())
