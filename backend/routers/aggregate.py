import os
import json
import asyncio
import anthropic
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from services.retrieval import get_all_tables
from typing import AsyncGenerator

load_dotenv()

router = APIRouter()

# Target ~3k tokens per batch to stay well under the 30k/min rate limit.
# Add an 8-second pause between batches → max ~7.5 batches/min × 3k = 22.5k tokens/min.
BATCH_CHAR_LIMIT = 12_000   # ~3k tokens at 4 chars/token
BATCH_DELAY_SECS = 8

EXTRACT_SYSTEM_PROMPT = """You are a precise financial data extraction assistant working with SEC filings.

Extract a specific value from the tables provided for each document.

STRICT RULES:
1. Only use numbers that appear explicitly in the tables. Never estimate or infer.
2. If the value is not found, respond with "Not found" for that document.
3. If multiple matching values exist in one document, list all of them.
4. Quote the exact cell text when citing a value.

Respond ONLY with a JSON array, no prose. Format:
[
  {"filename": "...", "page": N, "value": "...", "found": true},
  {"filename": "...", "page": null, "value": "Not found", "found": false}
]"""

AGGREGATE_SYSTEM_PROMPT = """You are a precise financial data aggregation assistant working with SEC filings.

You will be given extracted values per document. Produce a clean aggregation.

STRICT RULES:
1. Only sum values explicitly provided — never fill in missing ones.
2. Show the per-document breakdown table first, then the total.
3. Mark documents where the value was not found clearly.
4. End with a confidence note: found in N of M documents.

RESPONSE FORMAT:
## [Question]

### Breakdown by Document
| Document | Page | Value |
|---|---|---|
| filename | N | $X.X |
| filename | — | Not found |

### Total
**Total: $X.X** (found in N of M documents)

### Notes
- Any caveats or ambiguities here."""


class AggregateRequest(BaseModel):
    question: str
    doc_id: str | None = None


def _build_doc_context(filename: str, doc_chunks: list) -> str:
    parts = [f"=== {filename} ==="]
    for c in doc_chunks:
        parts.append(f"[Page {c.get('page', '?')}]\n{c['text']}")
    return "\n\n".join(parts)


def _batch_documents(doc_groups: dict[str, list]) -> list[list[tuple[str, list]]]:
    """Split documents into batches that stay within the char limit."""
    batches, current_batch, current_size = [], [], 0
    for filename, doc_chunks in doc_groups.items():
        doc_text = _build_doc_context(filename, doc_chunks)
        size = len(doc_text)
        if current_batch and current_size + size > BATCH_CHAR_LIMIT:
            batches.append(current_batch)
            current_batch, current_size = [], 0
        current_batch.append((filename, doc_chunks))
        current_size += size
    if current_batch:
        batches.append(current_batch)
    return batches


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
            yield f"data: {json.dumps({'type': 'status', 'message': 'Fetching all tables from library\u2026'})}\n\n"

            chunks = get_all_tables(doc_id=req.doc_id)
            if not chunks:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No table chunks found in the document library.'})}\n\n"
                return

            # Group by document
            doc_groups: dict[str, list] = {}
            for c in chunks:
                doc_groups.setdefault(c["filename"], []).append(c)

            doc_count = len(doc_groups)
            batches = _batch_documents(doc_groups)

            yield f"data: {json.dumps({'type': 'status', 'message': f'Processing {doc_count} documents in {len(batches)} batch(es)\u2026'})}\n\n"

            # Emit citations
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

            # Pass 1: extract value per document, batch by batch
            all_extractions: list[dict] = []
            for i, batch in enumerate(batches):
                if i > 0:
                    await asyncio.sleep(BATCH_DELAY_SECS)
                yield f"data: {json.dumps({'type': 'status', 'message': f'Extracting values — batch {i+1} of {len(batches)}\u2026'})}\n\n"
                context = "\n\n".join(_build_doc_context(fn, dc) for fn, dc in batch)
                resp = await client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    system=EXTRACT_SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": f"Tables:\n\n{context}\n\n---\n\nQuestion: {req.question}\n\nReturn JSON only."
                    }],
                )
                raw = resp.content[0].text.strip()
                # Strip markdown code fences if present
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                try:
                    batch_results = json.loads(raw)
                    all_extractions.extend(batch_results)
                except Exception:
                    # If JSON parse fails, record as unknown for these docs
                    for fn, _ in batch:
                        all_extractions.append({"filename": fn, "page": None, "value": "Parse error", "found": False})

            # Pass 2: final aggregation
            yield f"data: {json.dumps({'type': 'status', 'message': 'Aggregating results\u2026'})}\n\n"
            extractions_text = json.dumps(all_extractions, indent=2)
            async with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=AGGREGATE_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Question: {req.question}\n\n"
                        f"Extracted values per document:\n{extractions_text}"
                    ),
                }],
            ) as stream:
                async for text in stream.text_stream:
                    yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
