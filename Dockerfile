FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake the embedding model so cold starts don't download ~80MB
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY backend/ .

RUN chmod +x startup.sh

ENV CHROMA_PATH=/data/chromadb
ENV PDF_STORAGE_DIR=/data/uploaded_pdfs
ENV PROMPTS_DB_PATH=/data/prompts.db
ENV PARSED_JSON_DIR=/data/parsed_documents
ENV JOBS_STATE_FILE=/data/jobs_state.json

EXPOSE 8000

CMD ["./startup.sh"]
