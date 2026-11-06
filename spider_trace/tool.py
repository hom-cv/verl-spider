import asyncio
import os
import time
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

from spider_trace.events import emit
from spider_trace.sql import database_path, execute_query
from spider_trace.sql_protocol import format_observation
from spider_trace.limits import TOOL_MAX_ROWS, TOOL_RESULT_CHARS


class SQLiteTool(BaseTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        self.instances = {}

    async def create(self, instance_id=None, **kwargs):
        settings = kwargs.get("create_kwargs", kwargs)
        instance_id = instance_id or uuid4().hex
        db_id = settings["db_id"]
        path = database_path(os.environ["SPIDER_DB_ROOT"], db_id)
        self.instances[instance_id] = (db_id, path)
        return instance_id, ToolResponse()

    def run_query(self, path, parameters):
        return execute_query(path, parameters.get("sql"),
                             max_rows=TOOL_MAX_ROWS, max_chars=TOOL_RESULT_CHARS)

    async def execute(self, instance_id, parameters, **kwargs):
        db_id, path = self.instances[instance_id]
        sql = parameters.get("sql")
        start = time.perf_counter_ns()
        result = await asyncio.to_thread(self.run_query, path, parameters)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        emit("tool", tool_name=self.name, arguments=parameters,
             db_id=db_id, sql=sql, result=result, execution_ms=elapsed)
        return ToolResponse(text=format_observation(result)), 0.0, {}

    async def release(self, instance_id, **kwargs):
        self.instances.pop(instance_id, None)
