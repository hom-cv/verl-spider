import contextvars
import json
import os
import socket
import time
from pathlib import Path

trajectory = contextvars.ContextVar("trajectory", default=None)


def emit(event, **fields):
    root = Path(os.environ["TRACE_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "trajectory_id": trajectory.get(),
              "timestamp_ns": time.time_ns(), **fields}
    # Each process writes its own file; writes are synchronous within its event loop.
    with (root / f"events-{socket.gethostname()}-{os.getpid()}.jsonl").open("a") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
