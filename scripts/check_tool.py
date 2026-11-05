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
    parser = ToolParser.get_tool_parser("hermes", tokenizer)
    tools = {tool.name: tool for tool in
             initialize_tools_from_config(str(root / "config/tools.yaml"))}
    if "execute_sql" not in tools:
        raise RuntimeError(f"Tool registration failed: found {list(tools)}")
    print("PASS: execute_sql is registered", flush=True)

    sql = "SELECT 1 AS probe;"
    text = '<tool_call>\n' + json.dumps({
        "name": "execute_sql", "arguments": {"sql": sql}
    }) + '\n</tool_call>'
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
        result = json.loads(response.text)
        print("SQL result:", json.dumps(result), flush=True)
        if not result.get("ok") or result.get("rows") != [[1]]:
            raise RuntimeError("SQL tool did not return the expected probe result")
    finally:
        if instance_id is not None:
            await tool.release(instance_id)
        trajectory.reset(token)
    print("PASS: SQL tool executed the parsed call and returned [[1]]", flush=True)
    # Also validate schema discovery with actual registered tools and parser.
    async def discover(name, arguments):
        text = '<tool_call>' + json.dumps({"name": name, "arguments": arguments}) + '</tool_call>'
        _, parsed = await parser.extract_tool_calls(tokenizer.encode(text, add_special_tokens=False))
        if len(parsed) != 1 or parsed[0].name != name:
            raise RuntimeError(f"Discovery parser failed for {name}")
        tool = tools[name]
        instance_id, _ = await tool.create(create_kwargs={"db_id": sources["db_id"]})
        token = trajectory.set("synthetic-tool-check")
        try:
            response, _, _ = await tool.execute(instance_id, json.loads(parsed[0].arguments))
            result = json.loads(response.text)
            if not result.get("ok") or not result.get("tables"):
                raise RuntimeError(f"Discovery failed: {name}: {result}")
            print(f"PASS: {name}: {json.dumps(result)}", flush=True)
            return result
        finally:
            await tool.release(instance_id)
            trajectory.reset(token)

    tables = await discover("list_tables", {})
    await discover("get_table_schema", {"table_names": tables["tables"][:2]})
    print("Wiring check passed. This does not test whether the model chooses to call tools.")


if __name__ == "__main__":
    asyncio.run(main())
