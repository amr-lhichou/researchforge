from pathlib import Path
from sentence_transformers import SentenceTransformer


import numpy as np 


embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
)

def calculate(expression: str) -> str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"




def chunk_text(text:str,chunk_size: int = 500)-> list[str]:
    words = text.split()

    chunks = []
    for i in range(0,len(text),chunk_size):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
    return chunks



def build_index():
    data_path = Path(__file__).parent.parent/"data"/"knowledge.txt"
    text = data_path.read_text(encoding="utf-8")

    chunks = chunk_text(text)
    embeddings = embedding_model.encode(chunks)

    return chunks, embeddings

DOCUMENT_CHUNKS, DOUCMENT_EMBEDDINGS = build_index()




def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


def search_documents(query:str,top_k,int=3) -> str:
    
    query_enbedding = embedding_model.encode(query)
    scored_chunks = []
    for chunk, chunk_embedding in zip(
        DOCUMENT_CHUNKS,
        DOUCMENT_EMBEDDINGS
    ):
        score = cosine_similarity(
            query_enbedding,
            chunk_embedding
        )
        scored_chunks.append((score,chunk))
    scored_chunks.sort(
        key = lambda x: x[0],
        reverse=True
    )
    top_chunks = scored_chunks[:top_k]
    return "\n\n".join(
        chunk for score ,chunk in top_chunks
    )



