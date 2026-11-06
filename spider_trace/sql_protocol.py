"""Pure parsing and result formatting, independent of model libraries."""
import html
import json
import re


def extract_sql_call(text):
    # Reasoning may discuss tags; only an explicit action outside it executes.
    action = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if any(tag in action for tag in ("<think>", "<solution>", "<observation>", "```")):
        return None
    if action.count("<sql>") != 1 or action.count("</sql>") != 1:
        return None
    match = re.search(r"<sql>(.*?)</sql>", action, flags=re.S)
    return match.group(1).strip() if match and match.group(1).strip() else None


def format_observation(result, max_chars=5500):
    """Stable tabular rendering; leave room under veRL's 6000-character cap."""
    def cell(value):
        text = json.dumps(value, ensure_ascii=True) if not isinstance(value, str) else value
        return html.escape(text, quote=False).replace("|", "&#124;").replace("\n", "\\n").replace("\r", "\\r")

    if not result.get("ok"):
        body = "Error: " + cell(result.get("error", "Unknown SQL error"))
    else:
        columns = result["columns"]
        lines = ["| " + " | ".join(map(cell, columns)) + " |",
                 "| " + " | ".join("---" for _ in columns) + " |"]
        lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in result["rows"][:50])
        if not result["rows"]:
            lines.append("(0 rows)")
        if result.get("truncated") or len(result["rows"]) > 50:
            lines.append("[truncated: row or result-size limit]")
        body = "\n".join(lines)
    if len(body) > max_chars:
        body = body[:max_chars] + "\n[truncated: observation character limit]"
    return "<observation>\n" + body + "\n</observation>"
