import json
import os
import re
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from services.pdf_renderer import render_page, get_page_count
from services.chroma_client import get_collection

router = APIRouter()

PARSED_DIR = Path(os.getenv("PARSED_JSON_DIR", "./parsed_documents"))


@router.get("/{doc_id}/page/{page_num}")
def get_page_image(
    doc_id: str,
    page_num: int,
    hl_left: float = Query(None),
    hl_top: float = Query(None),
    hl_right: float = Query(None),
    hl_bottom: float = Query(None),
    zoom: float = Query(2.0),
):
    box = None
    if all(v is not None for v in [hl_left, hl_top, hl_right, hl_bottom]):
        box = {"left": hl_left, "top": hl_top, "right": hl_right, "bottom": hl_bottom}
    try:
        img = render_page(doc_id, page_num, highlight_box=box, zoom=zoom)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="PDF not found for this document")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return Response(content=img, media_type="image/png")


@router.get("/{doc_id}/page-count")
def page_count(doc_id: str):
    n = get_page_count(doc_id)
    if n == 0:
        raise HTTPException(status_code=404, detail="PDF not found")
    return {"page_count": n}


@router.get("/{doc_id}/page/{page_num}/chunks")
def get_page_chunks(doc_id: str, page_num: int):
    """Return all indexed chunks for a given page (page_num is 0-indexed)."""
    collection = get_collection()
    page_1idx = page_num + 1
    where = {"$and": [{"doc_id": doc_id}, {"page": page_1idx}]}
    result = collection.get(where=where, include=["documents", "metadatas"])
    chunks = []
    for cid, doc, meta in zip(
        result.get("ids", []),
        result.get("documents", []),
        result.get("metadatas", []),
    ):
        box_left = meta.get("box_left")
        box_top = meta.get("box_top")
        if box_left is None or box_top is None:
            continue  # skip chunks with no position data
        chunks.append({
            "chunk_id": cid,
            "chunk_index": meta.get("chunk_index"),
            "doc_id": meta.get("doc_id", doc_id),
            "page": meta.get("page", page_1idx),
            "text": doc[:400],
            "chunk_type": meta.get("chunk_type", "text"),
            "section_heading": meta.get("section_heading", ""),
            "box_left": box_left,
            "box_top": meta.get("box_top"),
            "box_right": meta.get("box_right", 1.0),
            "box_bottom": meta.get("box_bottom", 1.0),
        })
    # Sort top-to-bottom, left-to-right
    chunks.sort(key=lambda c: (c["box_top"], c["box_left"]))
    return {"chunks": chunks}


@router.get("/{doc_id}/sections")
def get_sections(doc_id: str):
    collection = get_collection()
    result = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
    metas = result.get("metadatas") or []
    if not metas:
        raise HTTPException(status_code=404, detail="Document not found")

    sections = _sections_from_json(doc_id, metas) or _sections_from_chroma(metas)
    return {"doc_id": doc_id, "sections": sections}


def _sections_from_json(doc_id: str, metas: list) -> list:
    filename = (metas[0].get("filename", "") if metas else "")
    safe = filename.rsplit(".", 1)[0].replace(" ", "_")
    files = list(PARSED_DIR.glob(f"{safe}__*.json")) if safe else []
    if not files:
        return []
    try:
        data = json.loads(files[0].read_text())
        chunks = data.get("chunks", [])
        sections = []
        for i, chunk in enumerate(chunks):
            ctype = chunk.get("type", "text")
            if ctype in ("heading", "section_header", "title", "header"):
                g = chunk.get("grounding", {})
                text = re.sub(r"<a [^>]+></a>\n?", "", chunk.get("markdown", "")).strip()
                if text:
                    sections.append({
                        "chunk_index": i,
                        "text": text[:150],
                        "page": g.get("page", 0),
                        "box": g.get("box", {}),
                        "type": ctype,
                    })
        return sections
    except Exception:
        return []


def _sections_from_chroma(metas: list) -> list:
    seen = {}
    for m in sorted(metas, key=lambda x: x.get("page", 0)):
        p = m.get("page", 1)
        if p not in seen:
            seen[p] = {
                "chunk_index": m.get("chunk_index", 0),
                "text": f"Page {p}",
                "page": p - 1,
                "box": {},
                "type": "page",
            }
    return list(seen.values())
