import os
import json

from dotenv import load_dotenv
from groq import Groq

from schemas import ResearchAnswer
from tools import calculate


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate."
                }
            },
            "required": ["expression"]
        }
    }
}


def generate_research_answer(question: str) -> ResearchAnswer:

    messages = [
        {
            "role": "system",
            "content": (
                "You are ResearchForge, a technical research assistant. "
                "Use available tools when useful."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # First call: let the model decide whether it needs a tool
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        tools=[calculator_tool],
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    if response_message.tool_calls:

        print("Tool call requested!")

        messages.append(response_message)

        for tool_call in response_message.tool_calls:

            function_name = tool_call.function.name

            function_args = json.loads(
                tool_call.function.arguments
            )

            if function_name == "calculate":
                tool_result = calculate(
                    function_args["expression"]
                )
            else:
                tool_result = "Unknown tool"

            print("Tool result:", tool_result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result
                }
            )

    # Ask the model for the final structured response
    messages.append(
        {
            "role": "system",
            "content": (
                "Return the final answer as JSON with exactly these fields: "
                "'answer' as a string and "
                "'key_points' as a list of strings."
            )
        }
    )

    final_response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        response_format={"type": "json_object"}
    )

    raw_output = final_response.choices[0].message.content

    return ResearchAnswer.model_validate_json(raw_output)