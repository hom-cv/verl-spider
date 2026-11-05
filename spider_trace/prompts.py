"""Versioned instructions shared by preparation and runtime rollout prompts."""

PROMPT_VERSION = "sql-tools-v2"
TOOL_PROTOCOL = """Mandatory SQL verification protocol:
Before submitting <final_sql>, you MUST call execute_sql and inspect its result.
Your first response must contain a brief plan followed by an actual tool call.
Use exactly this syntax, replacing the example SQL with your query:
<tool_call>
{"name":"execute_sql","arguments":{"sql":"SELECT COUNT(*) FROM singer;"}}
</tool_call>
Stop immediately after </tool_call> and wait for the real tool response.
Saying "let's execute" or writing a SQL code block does NOT execute a tool.
Do not emit <final_sql> in the same response as your first tool call.
After receiving the result, explain briefly what it shows. If it is an error,
revise the query and call execute_sql again. Only after a successful tool result
may you submit the verified query in <final_sql>...</final_sql>.
Never invent tool results. Do not repeat a successful call merely to use a tool."""


def with_tool_protocol(messages):
    """Copy messages and upgrade old prepared prompts without rewriting data."""
    result = [dict(message) for message in messages]
    for message in result:
        if message.get("role") == "system":
            if TOOL_PROTOCOL not in message["content"]:
                message["content"] += "\n\n" + TOOL_PROTOCOL
            break
    else:
        result.insert(0, {"role": "system", "content": TOOL_PROTOCOL})
    return result
