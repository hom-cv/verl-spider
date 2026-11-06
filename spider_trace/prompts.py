"""The single prompt source for both data preparation and live rollouts.

Adapted from the user-provided SkySQL example; SQL tags are executable actions.
"""
from spider_trace.sql import connect
from spider_trace.limits import MAX_ASSISTANT_TURNS, TOOL_MAX_ROWS

PROMPT_VERSION = "sql-paper-v5"
SYSTEM_TEMPLATE = """Task Overview:
You are a data science expert. Below, you are provided with a database schema and
a natural language question. Understand the schema and generate a valid SQL query
to answer the question within limited turns. Break down the problem, explain your
approach briefly, and generate the solution.

Database Engine: SQLite
Database Schema:
{db_details}
This schema describes tables, columns, primary keys, foreign keys, and relationships.
External Knowledge: {external_knowledge}
Question: {question}

Instructions:
1. Output only the information asked in the question. Include no extra SELECT columns.
2. Return all requested information, without missing or extra information.
3. Before the final query, analyze the question, check relevant relationships,
   verify your query with the SQL tool, and refine any errors using real results.
4. Only read-only queries are allowed. Use deterministic expressions and explicit
   ORDER BY with unique tie breakers for LIMIT. Do not use random or time functions.

Format:
1. Put a brief explanation inside <think>...</think> after receiving new information.
2. Call the SQL tool using a single <sql>your SQL</sql> block to explore or verify.
   This block executes SQL; a Markdown code block does not. Stop after </sql> and
   wait for the actual result. Never generate an observation yourself.
3. Tool output is a text table inside <observation>...</observation>, capped at
   {max_rows} rows and a character limit. Errors and truncation are explicitly indicated.
   Use the observation to refine the query when needed.
4. When no further exploration is needed or you reach the turn limit, provide
   the final SQL inside <solution>...</solution>. A solution block does not call
   the tool. There are at most {max_turns} assistant turns ({max_calls} calls plus a final answer),
   subject to token limits. Do not repeat queries just to produce calls.

------------------------ START OF EXAMPLE ------------------------
This is a format demonstration, not data from the current question's database.
Question: how many pigs are in the farm?
Database Schema:
Table: animals
- id (INTEGER, PRIMARY KEY)
- species (TEXT)
- age (INTEGER)
- name (TEXT)
<think>I will count animals whose species is pig and verify the query.</think>
<sql>SELECT COUNT(*) FROM animals WHERE species = 'pig';</sql>
<observation>
| COUNT(*) |
| --- |
| 12 |
</observation>
<think>The result contains 12 pigs. The verified query answers the question.</think>
<solution>SELECT COUNT(*) FROM animals WHERE species = 'pig';</solution>
------------------------ END OF EXAMPLE ------------------------
Now solve the actual question above using its database, not the example database.
"""


def build_messages(question, db_path, external_knowledge="None provided."):
    conn = connect(db_path)
    try:
        schema = "\n".join(row[0] for row in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"))
    finally:
        conn.close()
    return [{"role": "system", "content": SYSTEM_TEMPLATE.format(
        db_details=schema, external_knowledge=external_knowledge, question=question,
        max_rows=TOOL_MAX_ROWS, max_turns=MAX_ASSISTANT_TURNS,
        max_calls=MAX_ASSISTANT_TURNS - 1)},
        {"role": "user", "content": "Question: " + question}]


def question_from_messages(messages):
    """Accept both original schema-bearing rows and newer question-only rows."""
    users = [m["content"] for m in messages if m.get("role") == "user"]
    if len(users) != 1:
        raise ValueError("Expected one prepared question")
    content = users[0]
    if content.startswith("Schema:\n"):
        if "\n\nQuestion: " not in content:
            raise ValueError("Legacy prompt is missing its question separator")
        return content.split("\n\nQuestion: ", 1)[1]
    return content.removeprefix("Question: ")


def tool_bindings(db_id):
    return {"execute_sql": {"create_kwargs": {"db_id": db_id}}}
