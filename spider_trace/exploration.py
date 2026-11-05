"""Schema-free SQL exploration prompts and sample-scoped tool bindings."""

PROMPT_VERSION = "sql-explore-v4"
TOOL_NAMES = ("list_tables", "get_table_schema", "execute_sql")
SYSTEM = '''You are a data science expert exploring a SQLite database.
You receive a natural-language question, but no schema. Discover the tables and
columns using tools; do not guess names. Start with list_tables, then request
the relevant definitions with get_table_schema (you may request several tables).
Generate SQL answering exactly the question, with no extra columns. Execute the
SQL with execute_sql, inspect its actual result, and revise if necessary.
Use read-only deterministic SQL; for LIMIT use ORDER BY with unique tie breakers.
Do not use random or time functions. Do not repeat successful calls unnecessarily.

You may briefly explain your plan before each call. Emit one actual tool call per
response, in <tool_call> tags containing JSON, never Markdown code fences. Stop
after the call and wait for the real tool response. Only then interpret results.
Never invent schema or results. Submit verified SQL in <final_sql>...</final_sql>.
You have up to 11 tool-call turns plus a final answer turn, subject to token limits.

Example of the interaction format ONLY (the example database is not your database):
User: How many pigs are on the farm?
Assistant: First I will discover the tables.
<tool_call>
{"name":"list_tables","arguments":{}}
</tool_call>
Tool: {"ok":true,"tables":["animals"]}
Assistant: I will inspect the animals table before writing the query.
<tool_call>
{"name":"get_table_schema","arguments":{"table_names":["animals"]}}
</tool_call>
Tool: {"ok":true,"tables":[{"name":"animals","create_sql":"CREATE TABLE animals (id INTEGER PRIMARY KEY, species TEXT)"}]}
Assistant: I will count rows whose species is pig and verify the query.
<tool_call>
{"name":"execute_sql","arguments":{"sql":"SELECT COUNT(*) FROM animals WHERE species = 'pig';"}}
</tool_call>
Tool: {"ok":true,"columns":["COUNT(*)"],"rows":[[12]],"truncated":false}
Assistant: The query returned 12 pigs.
<final_sql>SELECT COUNT(*) FROM animals WHERE species = 'pig';</final_sql>
End of example. For the actual question, discover its database and generate your
own calls. The application supplies Tool messages; never generate them yourself.'''


def exploration_prompt(messages):
    """Replace legacy instructions and remove the schema from old prepared rows."""
    result = [{"role": "system", "content": SYSTEM}]
    for original in messages:
        if original.get("role") == "system":
            continue
        message = dict(original)
        content = message.get("content", "")
        if message.get("role") == "user" and content.startswith("Schema:\n"):
            if "\n\nQuestion: " not in content:
                raise ValueError("Cannot strip legacy schema: missing Question separator")
            message["content"] = "Question: " + content.split("\n\nQuestion: ", 1)[1]
        result.append(message)
    return result


def tool_bindings(db_id):
    return {name: {"create_kwargs": {"db_id": db_id}} for name in TOOL_NAMES}
