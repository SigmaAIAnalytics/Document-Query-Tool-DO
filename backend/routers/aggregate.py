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

EXTRACT_SYSTEM_PROMPT = """You are a precise financial data extraction assistant working with SEC filings.

Extract values from the tables provided. The question may ask for ONE value or MULTIPLE values (e.g. several ticker symbols or line items).

STRICT RULES:
1. Only use numbers that appear explicitly in the tables. Never estimate or infer.
2. Emit ONE JSON object per (document, item) pair. If the question asks for 5 tickers, emit up to 5 objects per document.
3. If a value is not found for a given item in a document, still emit an object with "found": false.
4. The "label" field names the specific item being extracted (e.g. the ticker symbol or metric name).
5. Quote the exact cell text in "value" when found.

Respond ONLY with a valid JSON array — no prose, no markdown fences. Format:
[
  {"filename": "...", "page": 3, "label": "MSTR", "value": "1,000", "found": true},
  {"filename": "...", "page": null, "label": "STRF", "value": "Not found", "found": false}
]"""

AGGREGATE_SYSTEM_PROMPT = """You are a precise financial data aggregation assistant working with SEC filings.

You will be given extracted values per (document, label) pair. Produce a clean aggregation grouped by label.

STRICT RULES:
1. Only sum values that were explicitly found — never fill in missing ones.
2. Strip commas and currency symbols before summing numeric strings (e.g. "1,000" → 1000).
3. Show a per-document breakdown table for each label, then a grand total row.
4. Mark documents where the value was not found clearly.
5. End each section with a confidence note: found in N of M documents.

RESPONSE FORMAT (repeat the ### section for each label):

## Results

### LABEL_NAME
| Document | Page | Value |
|---|---|---|
| filename | 3 | 1,000 |
| filename | — | Not found |

**Total: X** (found in N of M documents)

### Notes
- Any caveats, unit ambiguities, or duplicates here."""


class AggregateRequest(BaseModel):
    question: str
    doc_id: str | None = None


def _build_doc_context(filename: str, doc_chunks: list) -> str:
    parts = [f"=== {filename} ==="]
    for c in doc_chunks:
        parts.append(f"[Page {c.get('page', '?')}]\n{c['text']}")
    return "\n\n".join(parts)


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
            yield "data: " + json.dumps({'type': 'status', 'message': 'Fetching all tables from library…'}) + "\n\n"

            chunks = get_all_tables(doc_id=req.doc_id)
            if not chunks:
                yield "data: " + json.dumps({'type': 'error', 'message': 'No table chunks found in the document library.'}) + "\n\n"
                return

            # Group by document
            doc_groups: dict[str, list] = {}
            for c in chunks:
                doc_groups.setdefault(c["filename"], []).append(c)

            doc_count = len(doc_groups)
            yield "data: " + json.dumps({'type': 'status', 'message': f'Extracting values from {doc_count} documents…'}) + "\n\n"

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
            yield "data: " + json.dumps({'type': 'citations', 'citations': citations}) + "\n\n"

            # Build full context — all documents in one call
            context = "\n\n".join(
                _build_doc_context(fn, dc) for fn, dc in doc_groups.items()
            )

            # Pass 1: extract values (single call with retry)
            async def call_with_retry(messages, max_tokens, system, retries=4):
                delay = 10
                for attempt in range(retries):
                    try:
                        return await client.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=max_tokens,
                            system=system,
                            messages=messages,
                        )
                    except anthropic.APIStatusError as e:
                        if e.status_code in (429, 529) and attempt < retries - 1:
                            await asyncio.sleep(delay)
                            delay *= 2
                        else:
                            raise

            resp = await call_with_retry(
                messages=[{
                    "role": "user",
                    "content": f"Tables:\n\n{context}\n\n---\n\nQuestion: {req.question}\n\nReturn JSON only."
                }],
                max_tokens=4096,
                system=EXTRACT_SYSTEM_PROMPT,
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            try:
                all_extractions = json.loads(raw)
                if not isinstance(all_extractions, list):
                    raise ValueError("Expected JSON array")
            except Exception as parse_err:
                preview = raw[:300].replace("\n", " ")
                yield "data: " + json.dumps({'type': 'error', 'message': f'Extraction parse error: {parse_err} — raw: {preview}'}) + "\n\n"
                return

            # Pass 2: final aggregation (streaming)
            yield "data: " + json.dumps({'type': 'status', 'message': 'Aggregating results…'}) + "\n\n"
            extractions_text = json.dumps(all_extractions, indent=2)
            agg_delay = 10
            for attempt in range(4):
                try:
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
                            yield "data: " + json.dumps({'type': 'text', 'text': text}) + "\n\n"
                    break
                except anthropic.APIStatusError as e:
                    if e.status_code in (429, 529) and attempt < 3:
                        yield "data: " + json.dumps({'type': 'status', 'message': f'API busy, retrying in {agg_delay}s…'}) + "\n\n"
                        await asyncio.sleep(agg_delay)
                        agg_delay *= 2
                    else:
                        raise

            yield "data: [DONE]\n\n"

        except Exception as exc:
            yield "data: " + json.dumps({'type': 'error', 'message': str(exc)}) + "\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
