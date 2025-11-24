# AI_Newsletter/vectors.py
import chromadb
from typing import List, Dict, Any
from pathlib import Path

CHROMA_PATH = Path(__file__).parent / "chroma_store"
client = chromadb.PersistentClient(path=str(CHROMA_PATH))

collection = client.get_or_create_collection(name="articles")

def add_articles(articles: List[Dict[str, Any]]):
    """
    article dict keys:
    - id
    - text
    - embedding
    - metadata (title, url, user_email)
    """
    ids = [a["id"] for a in articles]
    docs = [a["text"] for a in articles]
    embeddings = [a["embedding"] for a in articles]
    metadatas = [a.get("metadata", {}) for a in articles]

    collection.add(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas,
    )

def search_similar(query_embedding: List[float], top_k: int = 5):
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
