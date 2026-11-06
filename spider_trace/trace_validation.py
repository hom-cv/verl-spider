"""Protocol validation, deliberately separate from answer correctness/reward."""
from collections import Counter

from spider_trace.sql_protocol import extract_solution


def validate_trajectory(events):
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
    solution = extract_solution(assistants[-1].get("text", "")) if assistants else None
    if solution is None:
        errors.append("missing final solution in last assistant turn")
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
    for event in events:
        if event["event"] == "complete":
            ids, mask = event.get("response_ids", []), event.get("response_mask", [])
            if not ids or len(ids) != len(mask) or any(m not in (0, 1) for m in mask):
                errors.append("invalid response token mask")
    return errors
