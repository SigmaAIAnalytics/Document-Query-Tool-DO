import uuid
import asyncio
import json
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from services.landing_ai import submit_job, poll_job, extract_chunks
from services.embedder import embed
from services.chroma_client import get_collection
from services.pdf_renderer import save_pdf

PARSED_DIR = Path(os.getenv("PARSED_JSON_DIR", "./parsed_documents"))
JOBS_FILE = Path(os.getenv("JOBS_STATE_FILE", "./jobs_state.json"))

router = APIRouter()


def _load_jobs() -> dict[str, dict]:
    if JOBS_FILE.exists():
        try:
            return json.loads(JOBS_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_jobs(jobs: dict) -> None:
    try:
        JOBS_FILE.write_text(json.dumps(jobs, indent=2))
    except Exception as exc:
        print(f"[documents] Warning: could not persist job state: {exc}")


# Job state persisted to disk so it survives server restarts/reloads
_jobs: dict[str, dict] = _load_jobs()


def _infer_filing_type(filename: str) -> str:
    name = filename.upper()
    if "8-K" in name or "8K" in name:
        return "8-K"
    if "10-K" in name or "10K" in name:
        return "10-K"
    return "Unknown"


async def _process_job(
    internal_id: str,
    landing_job_id: str,
    file_bytes: bytes,
    filename: str,
    company_name: str,
):
    """Background task: poll Landing.ai then index into ChromaDB."""
    try:
        _jobs[internal_id]["landing_job_id"] = landing_job_id
        _jobs[internal_id]["status"] = "processing"
        _save_jobs(_jobs)

        result = await poll_job(landing_job_id)

        # Persist raw Landing.ai JSON to disk
        PARSED_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = filename.rsplit(".", 1)[0].replace(" ", "_")
        json_path = PARSED_DIR / f"{safe_name}__{internal_id[:8]}.json"
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"[documents] Saved parsed JSON to {json_path}")

        chunks = extract_chunks(result)

        if not chunks:
            _jobs[internal_id] = {**_jobs[internal_id], "status": "error", "error": "No text extracted from PDF"}
            _save_jobs(_jobs)
            return

        filing_type = _infer_filing_type(filename)
        resolved_company = company_name.strip() or filename.rsplit(".", 1)[0]
        doc_id = internal_id

        texts = [c["text"] for c in chunks]
        embeddings = embed(texts)

        # Persist PDF to disk for page rendering
        save_pdf(doc_id, file_bytes)

        collection = get_collection()
        ids = [f"{doc_id}_chunk_{c['chunk_index']}" for c in chunks]
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "company_name": resolved_company,
                "filing_type": filing_type,
                "page": c["page"],
                "chunk_index": c["chunk_index"],
                "chunk_type": c.get("chunk_type", "text"),
                "section_heading": c.get("section_heading", ""),
                "char_count": c.get("char_count", len(c["text"])),
                "box_left": c.get("box_left", 0.0),
                "box_top": c.get("box_top", 0.0),
                "box_right": c.get("box_right", 1.0),
                "box_bottom": c.get("box_bottom", 1.0),
            }
            for c in chunks
        ]
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

        _jobs[internal_id] = {
            **_jobs[internal_id],
            "status": "done",
            "doc_id": doc_id,
            "filename": filename,
            "company_name": resolved_company,
            "filing_type": filing_type,
            "chunk_count": len(chunks),
            "parsed_json": str(json_path),
        }
        _save_jobs(_jobs)

    except Exception as exc:
        _jobs[internal_id] = {**_jobs[internal_id], "status": "error", "error": str(exc)}
        _save_jobs(_jobs)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    company_name: str = Form(default=""),
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    filename = file.filename or "document.pdf"

    try:
        landing_job_id = await submit_job(file_bytes, filename)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Landing.ai submit error: {exc}")

    internal_id = str(uuid.uuid4())
    _jobs[internal_id] = {"status": "processing", "filename": filename, "landing_job_id": landing_job_id}
    _save_jobs(_jobs)

    asyncio.create_task(_process_job(internal_id, landing_job_id, file_bytes, filename, company_name))

    return {"job_id": internal_id, "filename": filename, "status": "processing"}


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("")
def list_documents():
    collection = get_collection()
    result = collection.get(include=["metadatas"])

    seen: dict[str, dict] = {}
    for meta in result.get("metadatas") or []:
        doc_id = meta.get("doc_id", "")
        if doc_id and doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "filename": meta.get("filename", ""),
                "company_name": meta.get("company_name", ""),
                "filing_type": meta.get("filing_type", ""),
            }
    return list(seen.values())


@router.delete("/{doc_id}")
def delete_document(doc_id: str):
    collection = get_collection()
    result = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
    ids = result.get("ids") or []

    if not ids:
        raise HTTPException(status_code=404, detail="Document not found")

    collection.delete(ids=ids)
    return {"deleted": doc_id, "chunks_removed": len(ids)}


@router.get("/parsed")
def list_parsed_documents():
    """Return parsed JSONs on disk that are not currently indexed in ChromaDB."""
    if not PARSED_DIR.exists():
        return []

    collection = get_collection()
    result = collection.get(include=["metadatas"])
    indexed_filenames = {
        meta.get("filename", "") for meta in (result.get("metadatas") or [])
    }

    PDF_DIR = Path(os.getenv("PDF_STORAGE_DIR", "./uploaded_pdfs"))
    unindexed = []

    for json_file in sorted(PARSED_DIR.glob("*.json")):
        stem = json_file.stem  # e.g. "form-8-k_01-02-2026__e9c3b657"
        parts = stem.rsplit("__", 1)
        if len(parts) != 2:
            continue
        safe_name, id_prefix = parts
        original_filename = safe_name + ".pdf"

        if original_filename in indexed_filenames:
            continue

        # Find the matching PDF (stored as {full_uuid}.pdf)
        matching = list(PDF_DIR.glob(f"{id_prefix}*.pdf")) if PDF_DIR.exists() else []
        doc_id = matching[0].stem if matching else None

        unindexed.append({
            "json_filename": json_file.name,
            "filename": original_filename,
            "filing_type": _infer_filing_type(original_filename),
            "doc_id": doc_id,
            "has_pdf": doc_id is not None,
        })

    return unindexed


class ReindexRequest(BaseModel):
    json_filename: str
    company_name: str = ""


@router.post("/reindex")
def reindex_document(req: ReindexRequest):
    """Re-index a previously parsed document from its saved JSON — no Landing.ai call."""
    json_path = PARSED_DIR / req.json_filename
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Parsed JSON not found")

    try:
        data = json.loads(json_path.read_text())
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read JSON: {exc}")

    from services.landing_ai import extract_chunks
    chunks = extract_chunks(data)
    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks could be extracted from the parsed JSON")

    # Derive original filename and doc_id from the JSON filename
    stem = json_path.stem
    parts = stem.rsplit("__", 1)
    safe_name, id_prefix = (parts[0], parts[1]) if len(parts) == 2 else (stem, "")
    original_filename = safe_name + ".pdf"

    # Reuse the existing doc_id (UUID) so PDF page rendering still works
    PDF_DIR = Path(os.getenv("PDF_STORAGE_DIR", "./uploaded_pdfs"))
    matching = list(PDF_DIR.glob(f"{id_prefix}*.pdf")) if (id_prefix and PDF_DIR.exists()) else []
    doc_id = matching[0].stem if matching else str(uuid.uuid4())

    filing_type = _infer_filing_type(original_filename)
    company_name = req.company_name.strip() or safe_name

    texts = [c["text"] for c in chunks]
    embeddings = embed(texts)

    collection = get_collection()
    ids = [f"{doc_id}_chunk_{c['chunk_index']}" for c in chunks]
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": original_filename,
            "company_name": company_name,
            "filing_type": filing_type,
            "page": c["page"],
            "chunk_index": c["chunk_index"],
            "chunk_type": c.get("chunk_type", "text"),
            "section_heading": c.get("section_heading", ""),
            "char_count": c.get("char_count", len(c["text"])),
            "box_left": c.get("box_left", 0.0),
            "box_top": c.get("box_top", 0.0),
            "box_right": c.get("box_right", 1.0),
            "box_bottom": c.get("box_bottom", 1.0),
        }
        for c in chunks
    ]
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    return {
        "doc_id": doc_id,
        "filename": original_filename,
        "company_name": company_name,
        "filing_type": filing_type,
        "chunk_count": len(chunks),
    }
