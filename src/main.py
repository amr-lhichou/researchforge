import os

from dotenv import load_dotenv
from groq import Groq
from schemas import ResearchAnswer

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

question = input("Research question: ")
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are ResearchForge, a technical research assistant. "
                "Answer clearly and accurately. "
                "Return ONLY valid JSON with exactly these fields: "
                "'answer', which must be a string, and "
                "'key_points', which must be a list of strings."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]
    response_format = {"type"  :  "json_object"}
)
raw_output = response.choices[0].message.content
result = ResearchAnswer.model_validate_json(raw_output)


print("\nAnswer:")
print(result.answer)
print("\nKey points:")
for point in result.key_points:
    print(f"- {point}")