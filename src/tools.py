import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer


import numpy as np 


embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
)

client = chromadb.Client()
collection = client.get_or_create_collection(
    name="researchforge"
)



def chunk_text(text:str,chunk_size: int = 50)-> list[str]:
    words = text.split()

    chunks = []
    for i in range(0,len(words),chunk_size):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
    return chunks







def index_documents():
    data_path = Path(__file__).parent.parent/"data"/"knowledge.txt"
    text = data_path.read_text(encoding="utf-8")

    chunks = chunk_text(text)
    embeddings = embedding_model.encode(chunks)

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]
    metadatas = [
        {
            "source":"knowledge.txt",
            "chunk_id": i
        }
        for i in range(len(chunks))
    ]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )





def calculate(expression: str) -> str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"




"""def build_index():
    data_path = Path(__file__).parent.parent/"data"/"knowledge.txt"
    text = data_path.read_text(encoding="utf-8")

    chunks = chunk_text(text)
    embeddings = embedding_model.encode(chunks)

    return chunks, embeddings

DOCUMENT_CHUNKS, DOUCMENT_EMBEDDINGS = build_index()"""




def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


def search_documents(query:str,top_k,int=3) -> str:
    
    query_embedding = embedding_model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    documents = results["documents"][0]

    return "\n\n".join(documents)
    


