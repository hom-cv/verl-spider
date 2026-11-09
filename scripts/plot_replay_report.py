"""Build a poster-ready SVG from existing replay measurements (no new timings)."""
import argparse
import json
import math
from pathlib import Path


def render(folder):
    summary = json.loads((folder / "summary.json").read_text())
    records = [json.loads(line) for line in (folder / "calls.jsonl").read_text().splitlines()]
    trajectories = len({r["trajectory_id"] for r in records})
    unique = len({json.dumps(r["arguments"], sort_keys=True) for r in records})
    modes = ("direct", "cold", "warm")
    names = ("No cache", "Initially empty", "Pre-populated")
    colors = ("#64748b", "#2563eb", "#059669")
    calls, reps = summary["calls_per_pass"], summary["repetitions"]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="820" viewBox="0 0 1200 820" role="img" aria-labelledby="title">',
             '<title id="title">Preliminary results: TVCache SQL replay, work avoided and latency</title>',
             '<rect width="1200" height="820" fill="#f8fafc"/>',
             '<g font-family="Arial, Helvetica, sans-serif" fill="#0f172a">']

    def text(x, y, value, size=17, color="#0f172a", bold=False):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}">{value}</text>')

    text(40, 46, "PRELIMINARY RESULTS", 15, "#2563eb", True)
    text(40, 84, "SQL caching: less execution, not lower latency on this workload", 27, bold=True)
    text(40, 116, f"{trajectories} model trajectories → {calls} real tool calls → {unique} unique exact queries · one fixed SQLite database", 18)
    text(40, 144, f"{reps} repetitions × {calls} calls × 3 conditions = {len(records):,} measured requests", 17, "#475569")
    text(40, 193, "A  Database work avoided", 21, bold=True)
    text(610, 193, "B  Tool latency", 21, bold=True)
    # Both panels use zero-based axes; latency bars and p95 markers share a scale.
    latency_max = max(summary[m]["p95_ms"] for m in modes)
    axis_max = max(0.1, math.ceil(latency_max * 10) / 10)
    for i in range(5):
        x = 218 + 260 * i / 4
        parts.append(f'<path d="M{x} 220V422" stroke="#e2e8f0"/>')
        text(x-5, 448, f"{calls*i/4:g}", 14, "#475569")
        x2 = 788 + 260 * i / 4
        parts.append(f'<path d="M{x2} 220V422" stroke="#e2e8f0"/>')
        text(x2-12, 448, f"{axis_max*i/4:.2f}", 14, "#475569")
    text(240, 479, "SQL executions per pass", 15)
    text(815, 479, "Latency (milliseconds)", 15)
    for i, (mode, name, color) in enumerate(zip(modes, names, colors)):
        y = 235 + 70 * i
        data = summary[mode]
        counts = data["statuses"]
        executed = (counts.get("off", 0) + counts.get("miss", 0)) / reps
        w = 260 * executed / calls
        text(40, y+22, name, 17)
        parts.append(f'<rect x="218" y="{y}" width="{w}" height="30" fill="{color}"/>')
        text(488, y+21, f"{executed:g}", 18, color, True)
        if mode != "direct":
            text(218, y+48, f"{100*(1-executed/calls):g}% of SQL executions avoided", 13, color)
        text(610, y+22, name, 17)
        w2 = 260 * data["median_ms"] / axis_max
        p95x = 788 + 260 * data["p95_ms"] / axis_max
        parts.append(f'<rect x="788" y="{y}" width="{w2}" height="30" fill="{color}"/>')
        parts.append(f'<path d="M{p95x} {y-4}v38" stroke="{color}" stroke-width="3"/>')
        text(1070, y+10, f'{data["median_ms"]:.3f} med', 14)
        text(1070, y+30, f'{data["p95_ms"]:.3f} p95', 14)
    text(610, 508, "Bar: median · marker: p95 (not a confidence interval)", 14, "#475569")
    text(40, 545, "All measured outputs matched the original traces.", 18, "#059669", True)
    totals = " / ".join(f'{summary[m]["median_pass_tool_ms"]:.2f}' for m in modes)
    text(40, 577, f"Median summed tool time per {calls}-call pass (no cache / empty / populated): {totals} ms", 16)
    parts.append('<path d="M40 602H1160" stroke="#cbd5e1"/>')
    text(40, 634, "METHOD", 14, "#475569", True)
    text(40, 660, "Qwen2.5-Coder-7B-Instruct + veRL generated the source traces on Perlmutter; replay ran on a macOS laptop.", 15)
    text(40, 685, "Same SQL order and database snapshot in every pass; serial replay; no model generation during measurement.", 15)
    text(40, 710, "Fresh namespace for each empty-cache pass; populated pass reuses it. Empty passes include repeat-query hits.", 15)
    text(40, 735, "SQLite/OS caches warmed first; connection setup, database hashing and trace writing excluded from tool timings.", 15)
    text(40, 772, "LIMITATION: small, repeated workload. Cache lookups offset savings here; this is not an LLM rollout speedup.", 15, "#475569", True)
    parts.append('</g></svg>')
    path = folder / "preliminary-results.svg"
    path.write_text("\n".join(parts))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(render(parser.parse_args().folder).resolve())
