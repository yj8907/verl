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
        self.provider = config.get("provider") or ("anthropic" if self.model.startswith("claude") else "openai")

        if self.provider == "anthropic":
            api_key = os.environ["ANTHROPIC_API_KEY"]
            self.client = AsyncAnthropic(api_key=api_key, base_url=config.get("base_url"))
        elif self.provider == "openai":
            api_key = os.environ["OPENAI_API_KEY"]
            self.client = AsyncOpenAI(api_key=api_key, base_url=config.get("base_url"))
        else:
            raise ValueError(f"Unsupported provider '{self.provider}' for model '{self.model}'")

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating the instance.
        """
        if instance_id is None:
            return str(uuid4()), ToolResponse()
        else:
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


        return ToolResponse(text="Updated the tool state."), 0.0, {}
