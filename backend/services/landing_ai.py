import os
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

JOBS_URL = "https://api.va.landing.ai/v1/ade/parse/jobs"
POLL_INTERVAL = 10  # seconds between status checks
MAX_WAIT = 3600     # 60 minutes max


async def submit_job(file_bytes: bytes, filename: str) -> str:
    """Submit PDF to Landing.ai and return job_id immediately."""
    api_key = os.getenv("LANDING_AI_API_KEY")
    if not api_key:
        raise RuntimeError("LANDING_AI_API_KEY not set")

    headers = {"Authorization": f"Basic {api_key}"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            JOBS_URL,
            headers=headers,
            files={"document": (filename, file_bytes, "application/pdf")},
        )
        if not response.is_success:
            raise RuntimeError(f"Landing.ai submit error {response.status_code}: {response.text}")

        job_data = response.json()
        job_id = (
            job_data.get("job_id")
            or job_data.get("data", {}).get("job_id")
            or job_data.get("id")
        )
        if not job_id:
            raise RuntimeError(f"No job_id in Landing.ai response: {job_data}")

        return job_id


async def poll_job(job_id: str) -> dict:
    """Poll a job until complete. Returns the raw completed response dict."""
    api_key = os.getenv("LANDING_AI_API_KEY")
    headers = {"Authorization": f"Basic {api_key}"}
    status_url = f"{JOBS_URL}/{job_id}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        elapsed = 0
        while elapsed < MAX_WAIT:
            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

            poll = await client.get(status_url, headers=headers)
            if not poll.is_success:
                raise RuntimeError(f"Landing.ai poll error {poll.status_code}: {poll.text}")

            poll_data = poll.json()
            status = (
                poll_data.get("status")
                or poll_data.get("data", {}).get("status")
                or ""
            ).lower()
            progress = poll_data.get("progress", 0)

            print(f"[Landing.ai] job={job_id} status={status!r} progress={progress:.0%} elapsed={elapsed}s")

            if status in ("completed", "succeeded", "done", "success", "complete"):
                output_url = poll_data.get("output_url")
                if output_url:
                    output = await client.get(output_url)
                    if not output.is_success:
                        raise RuntimeError(f"Failed to fetch output: {output.status_code}: {output.text}")
                    return output.json()
                return poll_data

            if status in ("failed", "error", "cancelled"):
                reason = poll_data.get("failure_reason") or status
                raise RuntimeError(f"Landing.ai job failed: {reason}")

    raise RuntimeError(f"Landing.ai job {job_id} timed out after {MAX_WAIT}s")


# Chunk types that carry no useful text for RAG retrieval
_SKIP_TYPES = {"marginalia", "logo", "figure", "attestation"}

# Heuristics to detect heading-like text chunks (Landing.ai uses type=text for headings)
def _is_heading(text: str) -> bool:
    import re
    if text.startswith("#"):
        return True
    if re.match(r"^\*\*.+\*\*$", text.strip()):
        return True
    if re.match(r"^(Item\s+\d+[A-Z]?\.|PART\s+(I|II|III|IV|V))\b", text.strip(), re.IGNORECASE):
        return True
    return False


def extract_chunks(data: dict) -> list[dict]:
    # Unwrap envelope if results are nested under "data"
    if isinstance(data.get("data"), dict) and "chunks" in data["data"]:
        data = data["data"]

    raw_chunks = data.get("chunks") or []

    if raw_chunks and isinstance(raw_chunks, list):
        chunks = []
        current_heading = ""
        chunk_index = 0
        for i, chunk in enumerate(raw_chunks):
            chunk_type = chunk.get("type", "text")

            # Skip noise types
            if chunk_type in _SKIP_TYPES:
                continue

            text = chunk.get("markdown") or chunk.get("text") or chunk.get("content") or ""
            grounding = chunk.get("grounding") or {}
            page_num = grounding.get("page", i)
            if isinstance(page_num, (int, float)):
                page_num = int(page_num) + 1  # convert to 1-indexed
            else:
                page_num = i + 1

            clean_text = _strip_anchors(text)
            if not clean_text:
                continue

            # Track the current section heading
            if _is_heading(clean_text):
                current_heading = clean_text[:200]

            box = grounding.get("box") or {}
            chunks.append({
                "text": clean_text,
                "page": page_num,
                "chunk_index": chunk_index,
                "chunk_type": chunk_type,
                "section_heading": current_heading,
                "char_count": len(clean_text),
                "box_left": box.get("left", 0.0),
                "box_top": box.get("top", 0.0),
                "box_right": box.get("right", 1.0),
                "box_bottom": box.get("bottom", 1.0),
            })
            chunk_index += 1
        if chunks:
            return chunks

    # Fallback: split full markdown by double newline
    markdown = _strip_anchors(data.get("markdown") or "")
    if markdown:
        sections = [s.strip() for s in markdown.split("\n\n") if s.strip()]
        return [
            {"text": s, "page": i + 1, "chunk_index": i, "chunk_type": "text",
             "section_heading": "", "char_count": len(s)}
            for i, s in enumerate(sections)
        ]

    return [{"text": str(data), "page": 1, "chunk_index": 0, "chunk_type": "text",
             "section_heading": "", "char_count": len(str(data))}]


def _strip_anchors(text: str) -> str:
    import re
    return re.sub(r"<a [^>]+></a>\n?", "", text).strip()
