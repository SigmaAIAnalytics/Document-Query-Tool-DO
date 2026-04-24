import os
import json
import anthropic
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from services.retrieval import get_all_tables
from typing import AsyncGenerator

load_dotenv()

router = APIRouter()

AGGREGATE_SYSTEM_PROMPT = """You are a precise financial data aggregation assistant working with SEC filings.

You will be given ALL table chunks extracted from the document library. Your job is to find and aggregate specific values across documents.

STRICT RULES — follow these exactly, no exceptions:
1. Only use numbers that appear explicitly in the tables provided. Never estimate, infer, or hallucinate values.
2. If a value is not found in a document's tables, mark it as "Not found" — never assume it is zero.
3. Always present a source breakdown FIRST, showing each document and the exact value found, before any total.
4. Quote the exact cell text from the table when citing a value.
5. If the same value appears multiple times in one document (e.g. repeated header rows), count it only once.
6. If the question is ambiguous (e.g. multiple matching columns), list all candidates and ask the user to clarify rather than picking one.
7. End every response with a confidence note: how many documents contained the value vs. how many were checked.

RESPONSE FORMAT:
## [Question being answered]

### Breakdown by Document
| Document | Page | Value Found |
|---|---|---|
| filename | N | $X.X |
...

### Total
**[Total: $X.X]** (found in N of M documents)

### Notes
- Any caveats, ambiguities, or "not found" items explained here."""


class AggregateRequest(BaseModel):
    question: str
    doc_id: str | None = None


@router.post("/aggregate")
async def aggregate(req: AggregateRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate() -> AsyncGenerator[str, None]:
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Fetching all tables from library…'})}\n\n"

            chunks = get_all_tables(doc_id=req.doc_id)

            if not chunks:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No table chunks found in the document library.'})}\n\n"
                return

            doc_count = len(set(c["filename"] for c in chunks))
            yield f"data: {json.dumps({'type': 'status', 'message': f'Analysing {len(chunks)} table chunks across {doc_count} documents\u2026'})}\n\n"

            # Build context grouped by document
            doc_groups: dict[str, list] = {}
            for c in chunks:
                doc_groups.setdefault(c["filename"], []).append(c)

            context_parts = []
            for filename, doc_chunks in doc_groups.items():
                context_parts.append(f"=== {filename} ===")
                for c in doc_chunks:
                    context_parts.append(f"[Page {c.get('page', '?')}]\n{c['text']}")
            context = "\n\n".join(context_parts)

            # Emit citations for every table chunk
            citations = [
                {
                    "filename": c["filename"],
                    "page": c.get("page", 1),
                    "doc_id": c["doc_id"],
                    "page_0idx": int(c.get("page", 1)) - 1,
                    "filing_type": "",
                    "company_name": "",
                    "box_left": c.get("box_left"),
                    "box_top": c.get("box_top"),
                    "box_right": c.get("box_right"),
                    "box_bottom": c.get("box_bottom"),
                }
                for c in chunks
            ]
            yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"

            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Here are ALL the table chunks from the document library:\n\n"
                        f"{context}\n\n---\n\n"
                        f"Question: {req.question}"
                    ),
                }
            ]

            async with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=AGGREGATE_SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
