"""
Retrieval strategies used by the tool-use query router.
Each function returns a list of chunk dicts with text + metadata.
"""
from services.chroma_client import get_collection
from services.embedder import embed


def _where(doc_id: str | None, extra: dict | None = None) -> dict | None:
    conditions = []
    if doc_id:
        conditions.append({"doc_id": doc_id})
    if extra:
        conditions.append(extra)
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def _fmt(ids, docs, metas) -> list[dict]:
    out = []
    for cid, doc, meta in zip(ids, docs, metas):
        out.append({
            "chunk_id": cid,
            "text": doc,
            "page": meta.get("page"),
            "chunk_type": meta.get("chunk_type", "text"),
            "section_heading": meta.get("section_heading", ""),
            "filename": meta.get("filename", ""),
            "doc_id": meta.get("doc_id", ""),
            "box_left": meta.get("box_left"),
            "box_top": meta.get("box_top"),
            "box_right": meta.get("box_right"),
            "box_bottom": meta.get("box_bottom"),
        })
    return out


def search_semantic(query: str, doc_id: str | None = None, n: int = 8) -> list[dict]:
    col = get_collection()
    total = col.count()
    if total == 0:
        return []
    where = _where(doc_id)
    kwargs = dict(
        query_embeddings=[embed([query])[0]],
        n_results=min(n, total),
        include=["documents", "metadatas"],
    )
    if where:
        kwargs["where"] = where
    r = col.query(**kwargs)
    return _fmt(r["ids"][0], r["documents"][0], r["metadatas"][0])


def search_by_page(page: int, doc_id: str | None = None) -> list[dict]:
    col = get_collection()
    where = _where(doc_id, {"page": page})
    r = col.get(where=where, include=["documents", "metadatas"])
    return _fmt(r.get("ids", []), r.get("documents", []), r.get("metadatas", []))


def search_by_keyword(keyword: str, doc_id: str | None = None) -> list[dict]:
    col = get_collection()
    if col.count() == 0:
        return []
    where = _where(doc_id)
    kwargs = dict(
        where_document={"$contains": keyword},
        include=["documents", "metadatas"],
    )
    if where:
        kwargs["where"] = where
    r = col.get(**kwargs)
    return _fmt(r.get("ids", []), r.get("documents", []), r.get("metadatas", []))


def get_all_tables(doc_id: str | None = None) -> list[dict]:
    """Fetch every table chunk in the collection — exhaustive, no sampling."""
    col = get_collection()
    where = _where(doc_id, {"chunk_type": "table"})
    if where is None:
        where = {"chunk_type": "table"}
    r = col.get(where=where, include=["documents", "metadatas"])
    chunks = _fmt(r.get("ids", []), r.get("documents", []), r.get("metadatas", []))
    # Sort by document then page so context is ordered
    return sorted(chunks, key=lambda c: (c["filename"], c.get("page") or 0))


def search_similar_to_chunk(chunk_id: str, doc_id: str | None = None, n: int = 8) -> list[dict]:
    col = get_collection()
    src = col.get(ids=[chunk_id], include=["documents"])
    if not src["ids"]:
        return []
    embedding = embed([src["documents"][0]])[0]
    total = col.count()
    where = _where(doc_id)
    kwargs = dict(
        query_embeddings=[embedding],
        n_results=min(n + 1, total),
        include=["documents", "metadatas"],
    )
    if where:
        kwargs["where"] = where
    r = col.query(**kwargs)
    # Exclude the source chunk itself
    results = _fmt(r["ids"][0], r["documents"][0], r["metadatas"][0])
    return [c for c in results if c["chunk_id"] != chunk_id][:n]
