"""Protocol validation, deliberately separate from answer correctness/reward."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

from spider_trace.sql_protocol import extract_solution


@dataclass
class TraceValidation:
    integrity_errors: list[str]
    # None means infrastructure/integrity failure: do not score model protocol.
    protocol_errors: list[str] | None


def _tool_errors(event):
    """Validate recorded execution data, not whether the model's SQL succeeded."""
    errors = []
    for field in ("tool_name", "db_id"):
        if not isinstance(event.get(field), str) or not event[field]:
            errors.append(f"tool record missing valid {field}")
    arguments = event.get("arguments")
    if not isinstance(arguments, dict) or "sql" not in event or event["sql"] != arguments.get("sql"):
        errors.append("tool record has missing or inconsistent SQL arguments")
    elapsed = event.get("execution_ms")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        errors.append("tool record has invalid execution_ms")
    result = event.get("result")
    if not isinstance(result, dict) or type(result.get("ok")) is not bool:
        errors.append("tool record missing valid result.ok")
    elif result["ok"]:
        columns, rows = result.get("columns"), result.get("rows")
        if (not isinstance(columns, list) or not all(isinstance(c, str) for c in columns)
                or not isinstance(rows, list)
                or any(not isinstance(row, list) or len(row) != len(columns) for row in rows)
                or type(result.get("truncated")) is not bool):
            errors.append("tool record has malformed successful result")
    elif not isinstance(result.get("error"), str) or not result["error"]:
        errors.append("tool record missing SQL error details")
    return errors


def assess_trajectory(events):
    """Return integrity and protocol outcomes separately; never mutate events."""
    events = sorted(events, key=lambda event: event["timestamp_ns"])
    counts = Counter(event["event"] for event in events)
    errors = []
    if counts["start"] != 1 or counts["complete"] != 1:
        errors.append("expected exactly one start and one complete")
    if counts["error"]:
        errors.append("trajectory contains an error event")
    if not events or events[0]["event"] != "start" or events[-1]["event"] != "complete":
        errors.append("invalid trajectory event order")
    assistants = [e for e in events if e["event"] == "assistant"]
    if not assistants:
        errors.append("no assistant events")
    for event in events:
        if event["event"] == "tool":
            errors.extend(_tool_errors(event))
        if event["event"] == "complete":
            ids, mask = event.get("response_ids", []), event.get("response_mask", [])
            if not ids or len(ids) != len(mask) or any(m not in (0, 1) for m in mask):
                errors.append("invalid response token mask")
    if errors:
        return TraceValidation(errors, None)
    # Only intact traces can be evaluated for model protocol compliance.
    solution = extract_solution(assistants[-1].get("text", "")) if assistants else None
    if solution is None:
        errors.append("missing final solution or disallowed final-turn format")
    successful = [e for e in events if e["event"] == "tool"
                  and e.get("tool_name", "execute_sql") == "execute_sql"
                  and e.get("result", {}).get("ok") is True
                  and not e["result"].get("truncated", False)]
    if not successful:
        errors.append("no successful, untruncated SQL execution")
    elif solution is not None:
        # Do not claim semantic equivalence or rewrite SQL literals/identifiers.
        if not any(e.get("sql", "").strip() == solution
                   and e["timestamp_ns"] < assistants[-1]["timestamp_ns"] for e in successful):
            errors.append("final SQL does not exactly match a successful prior execution")
    return TraceValidation([], errors)


def validate_trajectory(events, *, require_verified_solution=True):
    """Compatibility helper; scripted wiring tests remain strict by default."""
    report = assess_trajectory(events)
    return report.integrity_errors + ((report.protocol_errors or []) if require_verified_solution else [])
