"""Pure parsing and result formatting, independent of model libraries."""
import html
import json
import re
from spider_trace.limits import OBSERVATION_BODY_CHARS, TOOL_MAX_ROWS


def close_assistant_turn(agent_data, eos_token_id):
    """Close a stopped tool-call turn without attributing the boundary to the model.

    Qwen uses its EOS token (<|im_end|>) as the chat message terminator.
    response_ids remains the unmodified model output for per-turn tracing.
    """
    if eos_token_id is None:
        raise ValueError("SQL tool turns require a tokenizer EOS/message-end token")
    if agent_data.prompt_ids and agent_data.prompt_ids[-1] == eos_token_id:
        return
    agent_data.prompt_ids.append(eos_token_id)
    agent_data.response_mask.append(0)
    if agent_data.response_logprobs:
        agent_data.response_logprobs.append(0.0)


def extract_sql_call(text):
    # Reasoning may discuss tags; only an explicit action outside it executes.
    action = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if any(tag in action for tag in ("<think>", "<solution>", "<observation>", "```")):
        return None
    if action.count("<sql>") != 1 or action.count("</sql>") != 1:
        return None
    match = re.search(r"<sql>(.*?)</sql>", action, flags=re.S)
    return match.group(1).strip() if match and match.group(1).strip() else None


def extract_solution(text):
    """Accept a single final answer outside explanations, not an example in them."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<think>" in text or "<sql>" in text or "<observation>" in text or "```" in text:
        return None
    matches = re.findall(r"<(solution|final_sql)>(.*?)</\1>", text, flags=re.S)
    if len(matches) != 1:
        return None
    return matches[0][1].strip() or None


def format_observation(result, max_chars=OBSERVATION_BODY_CHARS):
    """Stable tabular rendering; leave room under the configured response cap."""
    def cell(value):
        text = json.dumps(value, ensure_ascii=True) if not isinstance(value, str) else value
        return html.escape(text, quote=False).replace("|", "&#124;").replace("\n", "\\n").replace("\r", "\\r")

    if not result.get("ok"):
        body = "Error: " + cell(result.get("error", "Unknown SQL error"))
    else:
        columns = result["columns"]
        lines = ["| " + " | ".join(map(cell, columns)) + " |",
                 "| " + " | ".join("---" for _ in columns) + " |"]
        lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in result["rows"][:TOOL_MAX_ROWS])
        if not result["rows"]:
            lines.append("(0 rows)")
        if result.get("truncated") or len(result["rows"]) > TOOL_MAX_ROWS:
            lines.append("[truncated: row or result-size limit]")
        body = "\n".join(lines)
    if len(body) > max_chars:
        body = body[:max_chars] + "\n[truncated: observation character limit]"
    return "<observation>\n" + body + "\n</observation>"
