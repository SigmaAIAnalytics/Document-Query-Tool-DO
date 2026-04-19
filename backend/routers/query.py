import os
import json
import anthropic
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from services.retrieval import (
    search_semantic,
    search_by_page,
    search_by_keyword,
    search_similar_to_chunk,
)
from typing import AsyncGenerator

load_dotenv()

router = APIRouter()

SYSTEM_PROMPT = """You are a financial analyst assistant specializing in SEC filings (10-K and 8-K reports).
You have tools to search the document. Use them to find relevant content before answering.

ANCHOR CHUNK RULES (important):
- When the user message includes an [ANCHOR CHUNK], that is the primary source for the question.
- If the question can be answered directly from the anchor chunk content, answer from it WITHOUT calling any tools.
- Only call tools if the question explicitly asks to find the same information elsewhere ("where else", "cross-reference", "compare", "other sections").
- Never search for broader context when the user is asking specifically about the anchored content.

Tool guidance (for non-anchor or cross-reference questions):
- search_semantic: for conceptual/topical questions ("what are the risk factors?")
- search_by_page: when the user asks about a specific page number
- search_by_keyword: for finding exact numbers, stock tickers, or specific terms across the document
- search_similar_to_selection: when the user asks where else something appears in the document

Always cite sources: [Source: filename, page N].
Context includes [TEXT] and [TABLE] labels. Prefer tables for exact figures."""

TOOLS = [
    {
        "name": "search_semantic",
        "description": "Find document chunks semantically related to a query. Best for conceptual or topical questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "n": {"type": "integer", "description": "Number of results (default 8)", "default": 8},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_by_page",
        "description": "Retrieve all indexed chunks from a specific page number (1-indexed).",
        "input_schema": {
            "type": "object",
            "properties": {
                "page": {"type": "integer", "description": "1-indexed page number"},
            },
            "required": ["page"],
        },
    },
    {
        "name": "search_by_keyword",
        "description": "Find all chunks containing an exact keyword, number, or ticker symbol. Use for cross-referencing specific values.",
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Exact term or number to search for"},
            },
            "required": ["keyword"],
        },
    },
    {
        "name": "search_similar_to_selection",
        "description": "Find chunks similar to the user's currently selected chunk. Use when the user asks 'where else is this mentioned' or similar cross-reference questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of results (default 8)", "default": 8},
            },
            "required": [],
        },
    },
]


def _chunk_to_context(chunk: dict, idx: int) -> str:
    ctype = chunk.get("chunk_type", "text").upper()
    page = chunk.get("page", "?")
    filename = chunk.get("filename", "unknown")
    section = chunk.get("section_heading", "")
    section_label = f" | {section}" if section else ""
    header = f"[{ctype} | {filename} | Page {page}{section_label}]"
    return f"{header}\n{chunk['text']}"


def _chunk_to_citation(chunk: dict, idx: int) -> dict:
    page = chunk.get("page", 1)
    return {
        "excerpt_index": idx,
        "filename": chunk.get("filename", ""),
        "page": page,
        "company_name": "",
        "filing_type": "",
        "doc_id": chunk.get("doc_id", ""),
        "page_0idx": int(page) - 1 if page else 0,
        "box_left": chunk.get("box_left"),
        "box_top": chunk.get("box_top"),
        "box_right": chunk.get("box_right"),
        "box_bottom": chunk.get("box_bottom"),
    }


class QueryRequest(BaseModel):
    question: str
    doc_id: str | None = None
    anchor_chunk_ids: list[str] | None = None   # chunks selected by user in PDF viewer
    anchor_doc_ids: list[str] | None = None     # doc_ids the anchor chunks belong to


@router.post("/query")
async def query_documents(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate() -> AsyncGenerator[str, None]:
        try:
            all_chunks: dict[str, dict] = {}  # chunk_id -> chunk, deduplicated

            def _add_chunks(chunks: list[dict]):
                for c in chunks:
                    all_chunks[c["chunk_id"]] = c

            # Fetch all anchor chunks upfront so Claude sees their actual content
            anchor_ids = req.anchor_chunk_ids or []
            anchor_blocks = []
            if anchor_ids:
                from services.chroma_client import get_collection
                col = get_collection()
                anchor_result = col.get(ids=anchor_ids, include=["documents", "metadatas"])
                for cid, doc, meta in zip(
                    anchor_result.get("ids", []),
                    anchor_result.get("documents", []),
                    anchor_result.get("metadatas", []),
                ):
                    chunk_obj = {
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
                    }
                    all_chunks[cid] = chunk_obj
                    ctype = meta.get("chunk_type", "text").upper()
                    page = meta.get("page", "?")
                    filename = meta.get("filename", "")
                    anchor_blocks.append(
                        f"[ANCHOR CHUNK — {ctype} | {filename} | Page {page}]\n{doc}"
                    )

            # Build initial user message with all anchor content inline
            if anchor_blocks:
                anchors_text = "\n\n---\n\n".join(anchor_blocks)
                user_content = (
                    f"{anchors_text}\n\n"
                    f"[USER QUESTION]\n{req.question}"
                )
            else:
                user_content = req.question

            messages = [{"role": "user", "content": user_content}]

            # Tool-use loop (max 5 iterations)
            for iteration in range(5):
                response = await client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                )

                if response.stop_reason == "end_turn":
                    # Claude is done with tools — stream final answer
                    final_text = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            final_text = block.text
                            break

                    # Emit citations before text
                    citations = [
                        _chunk_to_citation(c, i + 1)
                        for i, c in enumerate(all_chunks.values())
                    ]
                    yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"

                    # Stream the final response via a new streaming call
                    context = "\n\n---\n\n".join(
                        _chunk_to_context(c, i) for i, c in enumerate(all_chunks.values())
                    )
                    final_messages = [
                        {
                            "role": "user",
                            "content": (
                                f"Context from SEC filings:\n\n{context}\n\n---\n\n"
                                f"Question: {req.question}"
                            ),
                        }
                    ]
                    async with client.messages.stream(
                        model="claude-sonnet-4-6",
                        max_tokens=2000,
                        system=SYSTEM_PROMPT,
                        messages=final_messages,
                    ) as stream:
                        async for text in stream.text_stream:
                            yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"

                    yield "data: [DONE]\n\n"
                    return

                if response.stop_reason != "tool_use":
                    break

                # Execute tool calls
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    tool_name = block.name
                    tool_input = block.input

                    yield f"data: {json.dumps({'type': 'status', 'message': f'Searching: {tool_name}...'})}\n\n"

                    try:
                        if tool_name == "search_semantic":
                            chunks = search_semantic(
                                tool_input["query"],
                                doc_id=req.doc_id,
                                n=tool_input.get("n", 8),
                            )
                        elif tool_name == "search_by_page":
                            chunks = search_by_page(tool_input["page"], doc_id=req.doc_id)
                        elif tool_name == "search_by_keyword":
                            chunks = search_by_keyword(tool_input["keyword"], doc_id=req.doc_id)
                        elif tool_name == "search_similar_to_selection":
                            if anchor_ids:
                                chunks = []
                                seen = set()
                                # Use anchor's own doc_id for scoping (fall back to query doc_id)
                                anchor_doc_map = {
                                    cid: all_chunks.get(cid, {}).get("doc_id") or req.doc_id
                                    for cid in anchor_ids
                                }
                                for aid in anchor_ids:
                                    target_doc = anchor_doc_map.get(aid) or req.doc_id
                                    for c in search_similar_to_chunk(
                                        aid, doc_id=target_doc, n=tool_input.get("n", 8)
                                    ):
                                        if c["chunk_id"] not in seen:
                                            chunks.append(c)
                                            seen.add(c["chunk_id"])
                            else:
                                chunks = []
                        else:
                            chunks = []

                        _add_chunks(chunks)
                        result_text = (
                            "\n\n---\n\n".join(_chunk_to_context(c, i) for i, c in enumerate(chunks))
                            if chunks else "No results found."
                        )
                    except Exception as exc:
                        result_text = f"Tool error: {exc}"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

            # Fallback if loop exhausted without end_turn
            if not all_chunks:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant content found.'})}\n\n"
                return

            citations = [
                _chunk_to_citation(c, i + 1) for i, c in enumerate(all_chunks.values())
            ]
            yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
