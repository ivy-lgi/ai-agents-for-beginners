import os
import asyncio
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core import (
    AgentId,
    CancellationToken,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
)
from autogen_agentchat.messages import TextMessage

load_dotenv()

deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")


class MyAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = AzureOpenAIChatCompletionClient(
            azure_deployment=deployment_name,
            model=deployment_name,
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        self._delegate = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_message(
        self, message: TextMessage, ctx: CancellationToken
    ) -> None:
        print(f"{self.id.type} received message: {message.content}")
        response = await self._delegate.on_messages(
            [TextMessage(content=message.content, source="user")],
            ctx.cancellation_token,
        )
        print(f"{self.id.type} responded: {response.chat_message.content}")


async def main():
    runtime = SingleThreadedAgentRuntime()
    await MyAgent.register(runtime, "my_agent", lambda: MyAgent("my_agent"))

    runtime.start()
    await runtime.send_message(
        TextMessage(content="Hello, World!", source="user"),
        AgentId("my_agent", "default"),
    )


if __name__ == "__main__":
    asyncio.run(main())
