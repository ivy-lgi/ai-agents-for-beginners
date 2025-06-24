import asyncio
from typing import Annotated
from dotenv import load_dotenv
import os
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.kernel import Kernel

load_dotenv()

deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]

chat_history = ChatHistory()
chat_history.add_user_message("I'd like to go to New York on January 1, 2025")


class BookTravelPlugin:
    @kernel_function(
        name="book_flight",
        description="Book a flight to a given location and on a given date",
    )
    async def book_flight(
        self,
        date: Annotated[str, "The date of the flight"],
        location: Annotated[str, "The location to fly to"],
    ) -> str:
        return f"Flight was booked to {location} on {date}"


kernel = Kernel()
kernel.add_plugin(BookTravelPlugin(), plugin_name="book_flight")

chat_service = AzureChatCompletion(
    deployment_name=deployment_name,
    api_key=api_key,
    endpoint=endpoint,
)

request_settings = AzureChatPromptExecutionSettings(
    function_choice_behavior=FunctionChoiceBehavior.Auto()
)


async def main():
    response = await chat_service.get_chat_message_content(
        chat_history=chat_history, settings=request_settings, kernel=kernel
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
