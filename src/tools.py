import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer


import numpy as np 


embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
)
CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"
client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)
collection = client.get_or_create_collection(
    name="researchforge"
)



def chunk_text(text:str,chunk_size: int = 50,overlap: int = 10)-> list[str]:
    words = text.split()

    chunks = []
    step = chunk_size - overlap
    for i in range(0,len(words),step):
        chunk = words[i:i+chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks







def index_documents():
    data_dir = Path(__file__).parent.parent/"data"
    for file_path in data_dir.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        embeddings = embedding_model.encode(chunks)

        ids = [
            f"{file_path.stem}_chunk_{i}"
            for i in range(len(chunks))
        ]
        metadatas = [
            {
                "source":file_path.name,
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
    retrieved_chunks = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    for document ,metadata in zip(documents,metadatas):
        retrieved_chunks.append(
            f"Source: {metadata['source']}\n"
            f"Chunk ID: {metadata['chunk_id']}\n"
            f"Content: {document}"
        )

    return "\n\n".join(retrieved_chunks)
    


