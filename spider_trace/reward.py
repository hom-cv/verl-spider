"""Small execution-match reward, not the official Spider evaluator."""
import json
import os
import re
from collections import Counter

from spider_trace.sql import database_path, execute_query


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    matches = re.findall(r"<final_sql>(.*?)</final_sql>", solution_str, flags=re.S)
    if not matches:
        return 0.0
    gold = json.loads(ground_truth)
    path = database_path(os.environ["SPIDER_DB_ROOT"], gold["db_id"])
    predicted = execute_query(path, matches[-1], max_rows=10000, max_chars=1000000)
    expected = execute_query(path, gold["sql"], max_rows=10000, max_chars=1000000)
    if not predicted["ok"] or not expected["ok"] or predicted["truncated"] or expected["truncated"]:
        return 0.0
    if len(predicted["columns"]) != len(expected["columns"]):
        return 0.0
    if gold["ordered"]:
        return float(predicted["rows"] == expected["rows"])
    return float(Counter(map(tuple, predicted["rows"])) == Counter(map(tuple, expected["rows"])))
