from pathlib import Path

def calculate(expression: str) -> str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def search_documents(query:str) -> str:
    data_path = Path(__file__).parent.parent/"data"/"knowledge.txt"
    text = data_path.read_text(encoding ="utf_8")
    paragraphs = text.split("\n\n")

    matches = []
    for paragraph in paragraphs :
        if query.lower() in paragraph.lower():
            matches.append(paragraph)
    if  not matches :
            return "No relevant info found."
    return "\n\n".join(matches)