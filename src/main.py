from llm import generate_research_answer


question = input("Research question: ")

result = generate_research_answer(question)

print("\nAnswer:")
print(result.answer)

print("\nKey points:")
for point in result.key_points:
    print(f"- {point}")

print("\nSources:")
for source in result.sources:
    print(f"-{source}")