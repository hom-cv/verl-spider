"""Versioned instructions shared by preparation and runtime rollout prompts."""

PROMPT_VERSION = "sql-tools-v3"
TOOL_PROTOCOL = """Mandatory SQL verification protocol:
Your task is to GENERATE AND EXECUTE SQL, not merely write a query.
1. Generate a read-only SQL query that answers the user's question using its schema.
2. In that SAME response, call execute_sql with the EXACT SQL you generated as the
   value of arguments.sql. Do not just describe the call or print a SQL code block.
3. Stop after the tool call and wait for the actual tool result.
4. Inspect the result. If the query failed or needs correction, generate revised
   SQL and call execute_sql with that revised SQL. After a successful result,
   submit the verified query in <final_sql>...</final_sql>.

Your first response MUST contain an actual <tool_call> block. A brief plan before
the block is optional. Do not wrap the block in Markdown backticks.
For example, ONLY IF your generated query is SELECT COUNT(*) FROM singer;, call:
<tool_call>
{"name":"execute_sql","arguments":{"sql":"SELECT COUNT(*) FROM singer;"}}
</tool_call>
For every other query, replace the example SQL with YOUR generated query. Use valid
JSON and escape any double quotes inside the SQL string. Never execute the example
query unless it answers the current question.
Stop immediately after </tool_call> and wait for the real tool response.
Saying "let's execute" or "I will run this" does NOT execute a tool. You must emit
the <tool_call> block containing the SQL, not end your response with that promise.
Do not emit <final_sql> in the same response as your first tool call.
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
