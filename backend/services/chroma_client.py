import os
import chromadb
from chromadb.config import Settings

_client = None
_collection = None

COLLECTION_NAME = "sec_filings"


def get_client() -> chromadb.Client:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_PATH", "/data/chromadb"),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection
