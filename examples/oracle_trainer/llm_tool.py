import os
from typing import Any, Optional
from uuid import uuid4

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

class LLMTool(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

        self.model = config["model"]
        self.max_tokens = config["max_tokens"]
        self.default_system_prompt = config.get("system_prompt", "")

        self.provider = config.get("provider") or ("anthropic" if self.model.startswith("claude") else "openai")

        if self.provider == "anthropic":
            api_key = os.environ["ANTHROPIC_API_KEY"]
            self.client = AsyncAnthropic(api_key=api_key, base_url=config.get("base_url"))
        elif self.provider == "openai":
            api_key = os.environ["OPENAI_API_KEY"]
            self.client = AsyncOpenAI(api_key=api_key, base_url=config.get("base_url"))
        else:
            raise ValueError(f"Unsupported provider '{self.provider}' for model '{self.model}'")

        # Per-instance conversation state: instance_id -> {"system_prompt": str, "messages": list[dict]}
        self._conversations: dict[str, dict[str, Any]] = {}

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.
            system_prompt: Per-trajectory override of the default system prompt.

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating the instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        system_prompt = kwargs.get("create_kwargs", {}).get("system_prompt", "")
        self._conversations[instance_id] = {
            "system_prompt": system_prompt if system_prompt is not None else self.default_system_prompt,
            "messages": [],
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the tool.

        Args:
            instance_id: The instance id of the tool.
            parameters: The json string of the parameters of the tool.

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The ToolResponse object containing text, image, and/or video content.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        conversation = self._conversations[instance_id]
        conversation["messages"].append({"role": "user", "content": parameters["content"]})

        if self.provider == "anthropic":
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=conversation["system_prompt"],
                messages=conversation["messages"],
            )
            text = response.content[0].text
        else:
            openai_messages = conversation["messages"]
            if conversation["system_prompt"]:
                openai_messages = [{"role": "system", "content": conversation["system_prompt"]}] + openai_messages
            response = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=openai_messages,
            )
            text = response.choices[0].message.content

        conversation["messages"].append({"role": "assistant", "content": text})
        return ToolResponse(text=text), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance and drop its stored conversation.

        Args:
            instance_id: The instance id of the tool.
        """
        self._conversations.pop(instance_id, None)
