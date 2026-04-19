import os
import json
import anthropic
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
from dotenv import load_dotenv
from services.prompts_db import save_prompt, list_prompts, get_prompt, delete_prompt, update_prompt, init_db
from services.embedder import embed
from services.chroma_client import get_collection

load_dotenv()
router = APIRouter()
init_db()

TOP_K = 8
SYSTEM_PROMPT = """You are a financial analyst assistant specializing in SEC filings (10-K and 8-K reports).
Answer questions using only the provided document excerpts. Be precise and cite your sources.

When citing sources, always include the document name and page number in this format: [Source: <filename>, page <page>]
If multiple chunks support an answer, cite all relevant sources.
If the provided context does not contain enough information to answer, say so clearly."""


class SavePromptRequest(BaseModel):
    name: str
    prompt_text: str
    doc_id: Optional[str] = None


class UpdatePromptRequest(BaseModel):
    name: str
    prompt_text: str


class RunPromptRequest(BaseModel):
    doc_id: Optional[str] = None


@router.get("")
def list_all_prompts(doc_id: Optional[str] = None):
    return list_prompts(doc_id=doc_id)


@router.post("")
def create_prompt(req: SavePromptRequest):
    return save_prompt(name=req.name, prompt_text=req.prompt_text, doc_id=req.doc_id)


@router.put("/{prompt_id}")
def edit_prompt(prompt_id: str, req: UpdatePromptRequest):
    updated = update_prompt(prompt_id, req.name, req.prompt_text)
    if not updated:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return updated


@router.delete("/{prompt_id}")
def remove_prompt(prompt_id: str):
    if not delete_prompt(prompt_id):
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"deleted": prompt_id}


@router.post("/{prompt_id}/run")
async def run_prompt(prompt_id: str, req: RunPromptRequest):
    prompt = get_prompt(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    doc_id = req.doc_id or prompt.get("doc_id")
    question = prompt["prompt_text"]

    collection = get_collection()
    total = collection.count()
    if total == 0:
        raise HTTPException(status_code=404, detail="No documents indexed")

    query_embedding = embed([question])[0]
    n_results = min(TOP_K, total)
    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if doc_id:
        kwargs["where"] = {"doc_id": doc_id}

    results = collection.query(**kwargs)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    context_parts, citations = [], []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        filename = meta.get("filename", "unknown")
        page = meta.get("page", "?")
        context_parts.append(f"[Excerpt {i+1} | {filename} | Page {page}]\n{doc}")
        citations.append({
            "excerpt_index": i + 1,
            "filename": filename,
            "page": page,
            "company_name": meta.get("company_name", ""),
            "filing_type": meta.get("filing_type", ""),
            "doc_id": meta.get("doc_id", ""),
            "page_0idx": int(page) - 1 if str(page).isdigit() else 0,
            "box_left": meta.get("box_left"),
            "box_top": meta.get("box_top"),
            "box_right": meta.get("box_right"),
            "box_bottom": meta.get("box_bottom"),
        })

    user_message = f"Context from SEC filings:\n\n{chr(10).join(context_parts)}\n\n---\n\nQuestion: {question}"
    client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate() -> AsyncGenerator[str, None]:
        try:
            yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
            async with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                async for text in stream.text_stream:
                    yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
