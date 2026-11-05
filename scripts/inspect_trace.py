import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
parser.add_argument("--trajectory", help="Print all events for this trajectory ID")
args = parser.parse_args()
groups = defaultdict(list)
for path in sorted(args.directory.glob("events-*.jsonl")):
    for line in path.read_text().splitlines():
        event = json.loads(line)
        groups[event["trajectory_id"]].append(event)
counts = Counter(e["event"] for group in groups.values() for e in group)
print(dict(counts))
tool_counts = Counter(e.get("tool_name", "execute_sql")
                      for events in groups.values() for e in events if e["event"] == "tool")
print("Tools:", dict(tool_counts))
for key, events in groups.items():
    print(key, "tool_calls=", sum(e["event"] == "tool" for e in events))
    if args.trajectory == key:
        for event in sorted(events, key=lambda e: e["timestamp_ns"]):
            print(json.dumps(event, indent=2))
if not counts["complete"] or not tool_counts["execute_sql"]:
    raise SystemExit("Trace check failed: expected completed trajectories and executed SQL (not just discovery)")
