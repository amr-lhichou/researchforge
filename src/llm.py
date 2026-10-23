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
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of relevant chunks to retrieve."
                }
            },
            "required": ["query"]
        }
    }
}


def generate_research_answer(question: str) -> ResearchAnswer:

    # 1. Retrieve relevant chunks from the knowledge base
    retrieved_context = search_documents(
        question,
        top_k=3
    )

    # 2. Build the prompt using the retrieved context
    messages = [
        {
            "role": "system",
            "content": (
                "You are ResearchForge, a technical research assistant. "
                "Answer strictly using the retrieved context provided by the user. "
                "Do not add information from your own general knowledge. "
                "If the retrieved context is insufficient, say so. "
                "Return JSON with exactly these fields: "
                "'answer' as a string and "
                "'key_points' as a list of strings."
            )
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Retrieved context:\n{retrieved_context}"
            )
        }
    ]

    # 3. Generate the grounded answer
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    # 4. Extract the JSON response
    raw_output = response.choices[0].message.content

    # 5. Validate it with Pydantic
    return ResearchAnswer.model_validate_json(raw_output)