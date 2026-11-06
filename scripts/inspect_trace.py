import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from spider_trace.trace_validation import assess_trajectory

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
parser.add_argument("--trajectory", help="Print all events for this trajectory ID")
parser.add_argument("--strict", action="store_true", help="Also fail on unsuccessful SQL protocol outcomes")
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
failures = 0
protocol_failures = 0
protocol_evaluated = 0
for key, events in groups.items():
    print(key, "tool_calls=", sum(e["event"] == "tool" for e in events))
    report = assess_trajectory(events)
    integrity_errors, protocol_errors = report.integrity_errors, report.protocol_errors
    protocol_evaluated += protocol_errors is not None
    protocol_failures += bool(protocol_errors)
    failures += bool(integrity_errors or (args.strict and protocol_errors))
    if integrity_errors:
        print("  INVALID TRACE (protocol not evaluated): " + "; ".join(integrity_errors))
    elif protocol_errors:
        print("  PRESERVED: model protocol unsuccessful: " + "; ".join(protocol_errors))
    else:
        print("  PASS: verified final SQL (not a correctness score)")
    if args.trajectory == key:
        for event in sorted(events, key=lambda e: e["timestamp_ns"]):
            print(json.dumps(event, indent=2))
if args.trajectory and args.trajectory not in groups:
    raise SystemExit("Requested trajectory was not found")
print(f"Protocol-unsuccessful trajectories: {protocol_failures}/{protocol_evaluated} evaluated; "
      f"{len(groups) - protocol_evaluated} not evaluated. No events removed or rewritten.")
if not groups or failures:
    raise SystemExit(f"Trace check failed: {failures}/{len(groups)} invalid trajectories (empty runs also fail)")
print(f"Trace check passed: {len(groups)} intact trajectories. This does not establish answer correctness.")
