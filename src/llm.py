import os
import json

from dotenv import load_dotenv
from groq import Groq

from schemas import ResearchAnswer
from tools import calculate,search_documents


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

search_documents_tool = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search the local knowledge base for relevant information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to use in the local documents."
                }
            },
            "required": ["query"]
        }
    }
}


def generate_research_answer(question: str) -> ResearchAnswer:

    messages = [
        {
            "role": "system",
            "content": (
            "You are ResearchForge, a technical research assistant. "
            "You have two custom tools available: "
            "calculate for mathematical expressions and "
            "search_documents for searching the local knowledge base. "
            "When the user asks about the knowledge base or local documents, "
            "use search_documents. Do not use browser search."
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
        tools=[calculator_tool,
        search_documents_tool],
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

            elif function_name == "search_documents":
                tool_result = search_documents(function_args["query"])
            else:
                tool_result = f"Unknown tool:{function_name}"

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
            "The tool results are already available above. "
            "Do not call any more tools. "
            "Do not use browser search. "
            "Using only the information already in the conversation, "
            "return the final answer as JSON with exactly these fields: "
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