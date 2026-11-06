import os
import time
from uuid import uuid4

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

from spider_trace.events import emit, trajectory
from spider_trace.prompts import (
    PROMPT_VERSION,
    build_messages,
    question_from_messages,
    tool_bindings,
)
from spider_trace.sql import database_path
from spider_trace.sql_protocol import close_assistant_turn
from spider_trace.sql_parser import (
    SQLBlockParser,  # Registers the sql parser before class initialization.
)


class TracedSQLAgent(ToolAgentLoop):
    async def run(self, sampling_params, **kwargs):
        info = kwargs.get("extra_info", {})
        path = database_path(os.environ["SPIDER_DB_ROOT"], info["db_id"])
        kwargs["raw_prompt"] = build_messages(question_from_messages(kwargs["raw_prompt"]), path)
        sampling_params = dict(sampling_params)
        # Delimit an action, without forcing the model to generate one.
        sampling_params.update(stop=["</sql>"], include_stop_str_in_output=True)
        token = trajectory.set(uuid4().hex)
        start = time.perf_counter()
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

    async def _handle_pending_state(self, agent_data, sampling_params):
        # Do not inject the tokenizer's JSON/Hermes tools template alongside SQL tags.
        agent_data.prompt_ids = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.apply_chat_template(
                agent_data.messages, add_generation_prompt=True, tokenize=True,
                **self.apply_chat_template_kwargs))
        if len(agent_data.prompt_ids) > self.prompt_length:
            raise ValueError("Rendered schema prompt exceeds data.max_prompt_length")
        return AgentState.GENERATING

    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        emit("assistant", turn=agent_data.assistant_turns,
             text=self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=False),
             next_state=state.name)
        if state == AgentState.PROCESSING_TOOLS:
            # The </sql> stop string cuts generation before Qwen's <|im_end|>.
            # Close that message before veRL appends the tool-response message.
            close_assistant_turn(agent_data, self.tokenizer.eos_token_id)
        return state
