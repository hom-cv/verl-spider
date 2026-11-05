import time
from uuid import uuid4

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

from spider_trace.events import emit, trajectory
from spider_trace.exploration import PROMPT_VERSION, exploration_prompt, tool_bindings


class TracedSQLAgent(ToolAgentLoop):
    async def run(self, sampling_params, **kwargs):
        kwargs["raw_prompt"] = exploration_prompt(kwargs["raw_prompt"])
        token = trajectory.set(uuid4().hex)
        start = time.perf_counter()
        info = kwargs.get("extra_info", {})
        kwargs["tools_kwargs"] = tool_bindings(info["db_id"])
        emit("start", prompt_version=PROMPT_VERSION,
             sample_index=int(info.get("index", -1)), db_id=info.get("db_id"),
             prompt=list(kwargs["raw_prompt"]), sampling=dict(sampling_params))
        try:
            output = await super().run(sampling_params, **kwargs)
            emit("complete", elapsed_s=time.perf_counter() - start,
                 prompt_text=self.tokenizer.decode(output.prompt_ids, skip_special_tokens=False),
                 response_text=self.tokenizer.decode(output.response_ids, skip_special_tokens=False),
                 response_ids=list(output.response_ids), response_mask=list(output.response_mask),
                 num_turns=output.num_turns)
            return output
        except Exception as exc:
            emit("error", error=str(exc))
            raise
        finally:
            trajectory.reset(token)

    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        emit("assistant", turn=agent_data.assistant_turns,
             text=self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=False),
             next_state=state.name)
        return state
