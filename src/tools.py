import chromadb
import hashlib
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

def compute_file_hash(file_path:Path) -> str:
    content = file_path.read_bytes()

    return hashlib.sha256(content).hexdigest()




def index_documents():
    data_dir = Path(__file__).parent.parent/"data"

    current_files = {
        file_path.name
        for file_path in data_dir.glob("*.txt")
    }

    stored = collection.get(
        include=["metadatas"]
    )

    stored_sources = {
        metadata["source"]
        for metadata in stored["metadatas"]
    }

    deleted_sources = stored_sources - current_files

    for source in deleted_sources:
        collection.delete(
            where={"source":source}
        )
        print(f"removed deleted file:{source}")
    for file_path in data_dir.glob("*.txt"):
        file_hash = compute_file_hash(file_path)
        

        existing = collection.get(
            where={
                "$and": [
                    {"source":file_path.name},
                    {"file_hash":file_hash}
                ]
            }
        )
        if existing["ids"]:
            print(f"skipping unchanged files:{file_path.name}")
            continue
        collection.delete(
            where={"source":file_path.name}
        )



        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        embeddings = embedding_model.encode(chunks)
        file_hash = compute_file_hash(file_path)

        ids = [
            f"{file_path.stem}_chunk_{i}"
            for i in range(len(chunks))
        ]
        metadatas = [
            {
                "source":file_path.name,
                "chunk_id": i,
                "file_hash":file_hash
            }
            for i in range(len(chunks))
        ]
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        print(f"Indexed:{file_path.name}")





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
    


