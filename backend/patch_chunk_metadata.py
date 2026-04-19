"""
Backfills chunk_type, section_heading, and char_count into existing ChromaDB chunks.
Run from backend/: python patch_chunk_metadata.py
"""
import json, re, os, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_SKIP_TYPES = {"marginalia", "logo", "figure", "attestation"}


def strip_anchors(text):
    return re.sub(r"<a [^>]+></a>\n?", "", text).strip()


def is_heading(text):
    if text.startswith("#"):
        return True
    if re.match(r"^\*\*.+\*\*$", text.strip()):
        return True
    if re.match(r"^(Item\s+\d+[A-Z]?\.|PART\s+(I|II|III|IV|V))\b", text.strip(), re.IGNORECASE):
        return True
    return False


def build_lookup(chunks):
    """Returns dict: clean_text -> {chunk_type, section_heading, char_count}"""
    lookup = {}
    current_heading = ""
    for chunk in chunks:
        chunk_type = chunk.get("type", "text")
        if chunk_type in _SKIP_TYPES:
            continue
        text = strip_anchors(chunk.get("markdown") or chunk.get("text") or "")
        if not text:
            continue
        if is_heading(text):
            current_heading = text[:200]
        lookup[text] = {
            "chunk_type": chunk_type,
            "section_heading": current_heading,
            "char_count": len(text),
        }
    return lookup


def main():
    from services.chroma_client import get_collection

    parsed_dir = Path(os.getenv("PARSED_JSON_DIR", "./parsed_documents"))
    json_files = list(parsed_dir.glob("*.json"))
    if not json_files:
        print("No parsed JSON files found in", parsed_dir)
        sys.exit(1)

    collection = get_collection()

    for json_path in json_files:
        print(f"\nProcessing {json_path.name}...")
        data = json.loads(json_path.read_text())
        lookup = build_lookup(data.get("chunks", []))
        print(f"  {len(lookup)} usable chunks in JSON")

        result = collection.get(include=["documents", "metadatas"])
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])

        updates_ids, updates_metas = [], []
        for cid, doc, meta in zip(ids, docs, metas):
            if meta.get("chunk_type") is not None:
                continue  # already patched
            clean = strip_anchors(doc)
            info = lookup.get(clean)
            if info:
                updates_ids.append(cid)
                updates_metas.append({**meta, **info})

        if not updates_ids:
            print("  No chunks needed patching.")
            continue

        batch = 500
        for i in range(0, len(updates_ids), batch):
            collection.update(
                ids=updates_ids[i:i+batch],
                metadatas=updates_metas[i:i+batch],
            )
        print(f"  Patched {len(updates_ids)} chunks.")

    print("\nDone.")


main()
