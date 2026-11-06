"""Check the existing parser and SQL tool without loading model weights."""
import asyncio
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict

from transformers import AutoTokenizer
from omegaconf import OmegaConf
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.utils.tool_registry import initialize_tools_from_config

from spider_trace.events import trajectory
from spider_trace.prompts import build_messages
from spider_trace.sql import database_path
from spider_trace.sql_parser import SQLBlockParser  # Registers the SQL-tag parser.
from spider_trace.limits import (MAX_ASSISTANT_TURNS, TOOL_RESPONSE_CHARS,
                                 PROMPT_TOKENS, RESPONSE_TOKENS)
from spider_trace.trace_validation import validate_trajectory


async def check_agent_flow(root, tokenizer, sources, output):
    """Real veRL state machine and SQLite; only model generation is scripted."""
    from spider_trace.agent import TracedSQLAgent
    from spider_trace.reward import compute_score

    config = OmegaConf.create({"data": {"apply_chat_template_kwargs": {}},
        "actor_rollout_ref": {"rollout": {
            "prompt_length": PROMPT_TOKENS, "response_length": RESPONSE_TOKENS,
            "multi_turn": {"max_user_turns": None,
                "max_assistant_turns": MAX_ASSISTANT_TURNS, "max_parallel_calls": 1,
                "max_tool_response_length": TOOL_RESPONSE_CHARS,
                "tool_response_truncate_side": "middle", "format": "sql",
                "tool_config_path": str(root / "config/tools.yaml"),
                "interaction_config_path": None}}}})

    class ScriptedServer:
        def __init__(self, empty_final):
            self.calls = 0
            self.empty_final = empty_final

        async def generate(self, *, prompt_ids, sampling_params, **kwargs):
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
            if self.calls == 0:
                assert "<tool_call>" not in prompt, "Conflicting JSON tool prompt"
                assert sampling_params["stop"] == ["</sql>"]
                text = "<sql>SELECT 1 AS probe;</sql>"
            elif self.calls == 1:
                first_turn = "<sql>SELECT 1 AS probe;</sql>"
                observation = "<observation>\n| probe |\n| --- |\n| 1 |\n</observation>"
                expected = tokenizer.apply_chat_template([
                    {"role": "system", "content": "SQL assistant."},
                    {"role": "user", "content": "Return one."},
                    {"role": "assistant", "content": first_turn},
                    {"role": "tool", "content": observation},
                ], tokenize=False, add_generation_prompt=True)
                assert prompt[prompt.rindex(first_turn):] == expected[expected.index(first_turn):], (
                    "Conversation boundary/observation differs from tokenizer chat template")
                assert "<observation>\n| probe |\n| --- |\n| 1 |" in prompt, "Missing actual result"
                assert prompt.endswith("<|im_start|>assistant\n"), "Missing assistant turn"
                text = ("" if self.empty_final else "<solution>SELECT 1 AS probe;</solution>") + tokenizer.eos_token
            else:
                raise AssertionError("Unexpected extra generation")
            self.calls += 1
            ids = tokenizer.encode(text, add_special_tokens=False)
            return SimpleNamespace(token_ids=ids, log_probs=[-0.1] * len(ids))

    for index, empty_final in ((-777, False), (-778, True)):
        server = ScriptedServer(empty_final)
        agent = TracedSQLAgent(SimpleNamespace(config=config), server, tokenizer, None)
        result = await agent.run({"temperature": 0.0},
            raw_prompt=[{"role": "user", "content": "Return the number one as probe."}],
            extra_info={"db_id": sources["db_id"], "index": index})
        assert server.calls == 2
        assert len(result.response_ids) == len(result.response_mask) == len(result.response_logprobs)
        generated = [token for token, mask in zip(result.response_ids, result.response_mask) if mask]
        first_turn = "<sql>SELECT 1 AS probe;</sql>"
        final_turn = ("" if empty_final else "<solution>SELECT 1 AS probe;</solution>") + tokenizer.eos_token
        expected = (tokenizer.encode(first_turn, add_special_tokens=False)
                    + tokenizer.encode(final_turn, add_special_tokens=False))
        assert generated == expected, "Synthetic boundary leaked into model mask"
        assert all(lp == 0.0 for lp, mask in zip(result.response_logprobs, result.response_mask) if not mask)
        text = tokenizer.decode(result.response_ids, skip_special_tokens=False)
        gold = json.dumps({"db_id": sources["db_id"], "sql": "SELECT 1 AS probe;", "ordered": False})
        assert compute_score("", text, gold) == (0.0 if empty_final else 1.0)

    groups = defaultdict(list)
    for path in output.glob("events-*.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            groups[event["trajectory_id"]].append(event)
    checked = 0
    for events in groups.values():
        starts = [e for e in events if e["event"] == "start"]
        if not starts:
            continue
        errors = validate_trajectory(events)
        if starts[0]["sample_index"] == -777:
            assert not errors, errors
            checked += 1
        elif starts[0]["sample_index"] == -778:
            assert any("missing final solution" in error for error in errors), errors
            checked += 1
    assert checked == 2
    print("PASS: scripted veRL SQL → real observation → final answer; empty-final regression rejected")


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
    await check_agent_flow(root, tokenizer, sources, output)
    print("Wiring check passed. This does not test whether the model chooses to call tools.")


if __name__ == "__main__":
    asyncio.run(main())
