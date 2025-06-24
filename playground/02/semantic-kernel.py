from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import (
    AzureChatCompletion,
)
from semantic_kernel.functions import KernelFunctionFromPrompt
from semantic_kernel.kernel import Kernel


async def main():

    kernel = Kernel()
    kernel.add_service(AzureChatCompletion())

    user_input = input("User Input:> ")

    kernel_function = KernelFunctionFromPrompt(
        function_name="SummarizeText",
        prompt="""
        Summarize the provided unstructured text in a sentence that is easy to understand.
        Text to summarize: {{$user_input}}
        """,
    )

    response = await kernel_function.invoke(kernel=kernel, user_input=user_input)
    print(f"Model Response: {response}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
