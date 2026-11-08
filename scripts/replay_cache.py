"""Serial fixed-trace SQL replay: direct, cold TVCache, warm TVCache.

No model generation. Cache-cold does not mean cold OS/SQLite page caches.
Connection setup, database hashing and correctness checks are outside timings.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import sqlite3
import statistics
import time
from uuid import uuid4

from tvclient import TVCacheClient
from spider_trace.cache import SQLCacheSession
from spider_trace.limits import TOOL_MAX_ROWS, TOOL_RESULT_CHARS
from spider_trace.sql import execute_query


def query(path, parameters):
    return execute_query(path, parameters.get("sql"), max_rows=TOOL_MAX_ROWS,
                         max_chars=TOOL_RESULT_CHARS)


def load_calls(run, database):
    sources = json.loads((run / "sources.json").read_text())
    actual_hash = hashlib.sha256(database.read_bytes()).hexdigest()
    if actual_hash != sources["database_sha256"]:
        raise ValueError("Database hash differs from original run")
    calls = []
    for path in sorted((run / "events").glob("events-*.jsonl")):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "tool":
                if event.get("tool_name") != "execute_sql" or event["db_id"] != sources["db_id"]:
                    raise ValueError("Replay supports only this run's single SQLite database")
                calls.append(event)
    if not calls:
        raise ValueError("No SQL calls found")
    return sorted(calls, key=lambda e: e["timestamp_ns"]), actual_hash


def p95(values):
    return sorted(values)[math.ceil(len(values) * 0.95) - 1]


def plot(summary, path):
    """Dependency-free SVG, generated from measured summary values."""
    modes = ("direct", "cold", "warm")
    colors = ("#64748b", "#2563eb", "#059669")
    maximum = max(summary[m]["p95_ms"] for m in modes) * 1.2
    scale = 550 / max(maximum, 0.001)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="510" viewBox="0 0 1000 510">',
           '<rect width="1000" height="510" fill="#f8fafc"/>',
           '<g font-family="Arial, sans-serif" fill="#0f172a">',
           '<text x="40" y="48" font-size="26" font-weight="bold">Fixed-trace SQL replay: tool latency</text>',
           f'<text x="40" y="78" font-size="16">{summary["repetitions"]} repetitions × {summary["calls_per_pass"]} calls per condition · lower is better</text>']
    for i, (mode, color) in enumerate(zip(modes, colors)):
        y = 120 + i * 90
        median, tail = summary[mode]["median_ms"], summary[mode]["p95_ms"]
        svg.extend([f'<text x="40" y="{y+25}" font-size="18">{mode.capitalize()}</text>',
                    f'<rect x="200" y="{y}" width="{median*scale:.2f}" height="36" fill="{color}"/>',
                    f'<path d="M{200+tail*scale:.2f} {y-5}v46" stroke="{color}" stroke-width="3"/>',
                    f'<text x="790" y="{y+15}" font-size="15">median {median:.3f} ms</text>',
                    f'<text x="790" y="{y+36}" font-size="15">p95 {tail:.3f} ms</text>'])
    svg.extend(['<text x="40" y="410" font-size="15">Bar = median; vertical marker = p95. Cold pass includes repeat-query hits.</text>',
                '<text x="40" y="438" font-size="15">Serial replay on this laptop; setup excluded; all results checked against source traces.</text>',
                '<text x="40" y="466" font-size="15">Not an LLM rollout speedup. SQLite/OS caches warmed before measurement.</text>',
                '</g></svg>'])
    path.write_text("\n".join(svg))


def plot_executions(summary, path):
    """Show actual work avoided, independently of latency; zero-based scale."""
    repetitions = summary["repetitions"]
    calls = summary["calls_per_pass"]
    modes = ("direct", "cold", "warm")
    colors = ("#64748b", "#2563eb", "#059669")
    counts = []
    for mode in modes:
        statuses = summary[mode]["statuses"]
        if sum(statuses.values()) != repetitions * calls:
            raise ValueError("Incomplete replay counts")
        counts.append((statuses.get("off", 0) + statuses.get("miss", 0)) / repetitions)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="450" viewBox="0 0 900 450" role="img" aria-labelledby="title">',
           '<title id="title">SQL executions avoided through caching, not latency speedup</title>',
           '<rect width="900" height="450" fill="#f8fafc"/>',
           '<g font-family="Arial, sans-serif" fill="#0f172a">',
           '<text x="36" y="44" font-size="25" font-weight="bold">Caching eliminates repeated SQL execution</text>',
           f'<text x="36" y="72" font-size="15" fill="#475569">Mean executions per {calls}-call pass · {repetitions} repetitions per condition</text>']
    for tick in range(5):
        value = calls * tick / 4
        x = 180 + 480 * tick / 4
        svg.append(f'<path d="M{x} 95V303" stroke="#e2e8f0"/>')
        svg.append(f'<text x="{x}" y="327" text-anchor="middle" font-size="14">{value:g}</text>')
    for i, (mode, color, count) in enumerate(zip(modes, colors, counts)):
        y = 110 + i * 70
        width = 480 * count / calls
        avoided = 100 * (1 - count / counts[0]) if counts[0] else 0
        svg.append(f'<text x="36" y="{y+25}" font-size="18">{mode.capitalize()}</text>')
        if count:
            svg.append(f'<rect x="180" y="{y}" width="{width}" height="38" rx="3" fill="{color}"/>')
        else:
            svg.append(f'<path d="M180 {y}v38" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{192+width}" y="{y+25}" font-size="18" font-weight="bold">{count:g}</text>')
        if mode != "direct":
            svg.append(f'<text x="720" y="{y+25}" font-size="16" fill="{color}">{avoided:g}% avoided</text>')
    svg.append('<text x="420" y="352" text-anchor="middle" font-size="14">SQL executions (cache lookups excluded)</text>')
    if summary.get("results_equal"):
        svg.append(f'<text x="36" y="391" font-size="15">All {repetitions*calls*3:,} measured results matched the original traces.</text>')
    svg.extend(['<text x="36" y="419" font-size="14" fill="#475569">Work avoided is not speedup. See companion latency chart for measured runtime.</text>',
                '</g></svg>'])
    path.write_text("\n".join(svg))


async def replay(args):
    calls, db_hash = load_calls(args.run, args.database)
    # Warm thread/SQLite caches and verify local execution against original results.
    # Strict order equality intentionally catches ordering/version differences.
    for event in calls:
        actual = await asyncio.to_thread(query, args.database, event["arguments"])
        if actual != event["result"]:
            raise ValueError(f"Local result differs from original trace: {event['sql']}")
    run_id = uuid4().hex
    output = args.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    records = []
    print(f"Results: {output.resolve()}", flush=True)

    async def measure(mode, repetition, session, stream):
        for index, event in enumerate(calls):
            start = time.perf_counter_ns()
            if mode == "direct":
                result = await asyncio.to_thread(query, args.database, event["arguments"])
                metadata = {"status": "off"}
            else:
                result, metadata = await session.execute(event["arguments"], query)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            equal = result == event["result"]
            record = {"mode": mode, "repetition": repetition, "call_index": index,
                      "trajectory_id": event["trajectory_id"], "arguments": event["arguments"],
                      "cache": metadata, "elapsed_ms": elapsed_ms,
                      "result_equal": equal, "result": result}
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if not equal:
                raise ValueError("Replay result mismatch; partial measurements retained")
            records.append(record)

    with (output / "calls.jsonl").open("w") as stream:
        for repetition in range(args.repetitions):
            async with TVCacheClient(host=args.host, port=args.port, retries=0) as client:
                session = SQLCacheSession(args.database, client=client,
                                          namespace=f"replay-{run_id}-{repetition}")
                # Cold/warm remain paired; alternate direct placement to reduce drift.
                modes = ("direct", "cold", "warm") if repetition % 2 == 0 else ("cold", "warm", "direct")
                for mode in modes:
                    await measure(mode, repetition, session, stream)
    summary = {"repetitions": args.repetitions, "calls_per_pass": len(calls),
               "database_sha256": db_hash, "sqlite_version": sqlite3.sqlite_version,
               "platform": platform.platform(), "source_run": str(args.run.resolve()),
               "concurrency": 1, "setup_timed": False, "results_equal": True}
    for mode in ("direct", "cold", "warm"):
        selected = [r for r in records if r["mode"] == mode]
        times = [r["elapsed_ms"] for r in selected]
        passes = [sum(r["elapsed_ms"] for r in selected if r["repetition"] == i)
                  for i in range(args.repetitions)]
        summary[mode] = {"median_ms": statistics.median(times), "p95_ms": p95(times),
                         "median_pass_tool_ms": statistics.median(passes),
                         "total_tool_ms": sum(times),
                         "statuses": dict(Counter(r["cache"]["status"] for r in selected))}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(summary, output / "latency.svg")
    plot_executions(summary, output / "executions.svg")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7878)
    parser.add_argument("--output-root", type=Path, default=Path("runs/cache-replay"))
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    asyncio.run(replay(args))
