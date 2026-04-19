"""
Patches bounding box metadata into existing ChromaDB chunks from saved parsed JSON.
Run from backend/: python patch_box_metadata.py
"""
import json, re, os, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def strip_anchors(text):
    return re.sub(r"<a [^>]+></a>\n?", "", text).strip()

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
        chunks = data.get("chunks", [])

        # Build lookup: clean_text -> box (using first match)
        text_to_box = {}
        for chunk in chunks:
            text = strip_anchors(chunk.get("markdown") or chunk.get("text") or "")
            grounding = chunk.get("grounding") or {}
            box = grounding.get("box") or {}
            if text and box:
                text_to_box[text] = box

        print(f"  {len(text_to_box)} chunks with box data in JSON")

        # Fetch all chunks from ChromaDB for this filename
        safe_name = json_path.stem.rsplit("__", 1)[0]
        filename_hint = safe_name.replace("_", "-") + ".pdf"

        result = collection.get(include=["documents", "metadatas"])
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])

        updates_ids, updates_metas = [], []
        matched = 0
        for cid, doc, meta in zip(ids, docs, metas):
            if meta.get("box_left") is not None:
                continue  # already has box data
            clean = strip_anchors(doc)
            box = text_to_box.get(clean)
            if box:
                new_meta = {
                    **meta,
                    "box_left": box.get("left", 0.0),
                    "box_top": box.get("top", 0.0),
                    "box_right": box.get("right", 1.0),
                    "box_bottom": box.get("bottom", 1.0),
                }
                updates_ids.append(cid)
                updates_metas.append(new_meta)
                matched += 1

        if not updates_ids:
            print("  No chunks needed patching.")
            continue

        # ChromaDB update in batches of 500
        batch = 500
        for i in range(0, len(updates_ids), batch):
            collection.update(
                ids=updates_ids[i:i+batch],
                metadatas=updates_metas[i:i+batch],
            )
        print(f"  Patched {matched} chunks with bounding box data.")

    print("\nDone.")

main()
