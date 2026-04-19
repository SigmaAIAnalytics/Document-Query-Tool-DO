#!/bin/bash
set -e

# On first boot, seed /data from the Docker image's baked-in data
if [ ! -f "/data/.seeded" ]; then
    echo "[startup] First boot — seeding /data from image..."
    mkdir -p /data/chromadb /data/uploaded_pdfs /data/parsed_documents

    [ -d "/app/chroma_data" ]     && cp -r /app/chroma_data/.     /data/chromadb/
    [ -d "/app/uploaded_pdfs" ]   && cp -r /app/uploaded_pdfs/.   /data/uploaded_pdfs/
    [ -d "/app/parsed_documents" ] && cp -r /app/parsed_documents/. /data/parsed_documents/
    [ -f "/app/prompts.db" ]       && cp /app/prompts.db            /data/prompts.db

    touch /data/.seeded
    echo "[startup] Seeding complete."
fi

exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
