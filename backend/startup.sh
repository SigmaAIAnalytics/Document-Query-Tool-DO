#!/bin/bash
set -e

# Sync baked-in data from the image into the persistent /data volume.
# Runs on every startup so new documents added to the image are picked up
# on each deployment without needing to wipe the volume.

mkdir -p /data/chromadb /data/uploaded_pdfs /data/parsed_documents

# ChromaDB must be replaced as a unit (SQLite + vector files must stay in sync)
if [ -d "/app/chroma_data" ]; then
    echo "[startup] Syncing ChromaDB from image..."
    rm -rf /data/chromadb
    cp -r /app/chroma_data/. /data/chromadb/
fi

# PDFs and parsed JSONs: only copy files not already on the volume
if [ -d "/app/uploaded_pdfs" ]; then
    echo "[startup] Syncing uploaded PDFs..."
    cp -n /app/uploaded_pdfs/*.pdf /data/uploaded_pdfs/ 2>/dev/null || true
fi

if [ -d "/app/parsed_documents" ]; then
    echo "[startup] Syncing parsed documents..."
    cp -n /app/parsed_documents/*.json /data/parsed_documents/ 2>/dev/null || true
fi

if [ -f "/app/prompts.db" ] && [ ! -f "/data/prompts.db" ]; then
    cp /app/prompts.db /data/prompts.db
fi

echo "[startup] Data sync complete."

exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
