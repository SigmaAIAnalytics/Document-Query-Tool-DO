# SEC Filings RAG

Upload and query SEC 10-K and 8-K filings using Retrieval-Augmented Generation.

**Stack:** FastAPI · Landing.ai ADE · ChromaDB · Anthropic Claude · React · Tailwind CSS

---

## Setup

### 1. Clone and configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys:
#   LANDING_AI_API_KEY=...
#   ANTHROPIC_API_KEY=...
```

### 2. Backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend

```bash
cd frontend
npm install
```

### 4. Start both servers

From the project root, run:

```bash
./start.sh
```

This launches the backend and frontend together. Press `Ctrl+C` to stop both.

| Service  | URL |
|----------|-----|
| Frontend | http://localhost:5173 |
| Backend  | http://localhost:8000 |
| API docs | http://localhost:8000/docs |

> **Manual start** — if you prefer to run them separately:
> ```bash
> # Terminal 1
> cd backend && source .venv/bin/activate && uvicorn main:app --reload --port 8000
> # Terminal 2
> cd frontend && npm run dev
> ```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/documents/upload` | Upload a PDF filing (`file`, optional `company_name`) |
| GET | `/documents` | List all indexed documents |
| DELETE | `/documents/{doc_id}` | Remove a document and all its chunks |
| GET | `/documents/jobs/{job_id}` | Poll upload job status |
| POST | `/query` | Ask a question — returns SSE stream with citations |
| GET | `/pages/{doc_id}/page/{n}` | Render page N as PNG (optional highlight box params) |
| GET | `/pages/{doc_id}/page-count` | Total pages in a document |
| GET | `/pages/{doc_id}/sections` | Section/TOC list for a document |
| GET | `/prompts` | List saved prompts |
| POST | `/prompts` | Save a new prompt |
| DELETE | `/prompts/{id}` | Delete a prompt |
| POST | `/prompts/{id}/run` | Run a saved prompt — returns SSE stream |

## Usage

1. **Upload** a 10-K or 8-K PDF via the drag-and-drop panel — Landing.ai parses it in the background
2. **Ask questions** in the chat panel — Claude answers with page-level citations
3. **Click any citation** to open the PDF viewer at the exact source page, with the relevant region highlighted
4. **Browse sections** — select a document in the library to expand its table of contents; click any section to jump to that page
5. **Save prompts** — click 💾 next to the input to name and save a question; run it again any time from the Saved Prompts panel
6. **Scope queries** — click a document in the library to filter all queries to that filing only

## Notes

- ChromaDB data is persisted in `backend/chroma_data/`
- Raw Landing.ai parsed JSON is saved to `backend/parsed_documents/`
- Uploaded PDFs are stored in `backend/uploaded_pdfs/` for page rendering
- Filing type (10-K / 8-K) is inferred from the filename; name files accordingly
- The `all-MiniLM-L6-v2` embedding model is downloaded automatically on first run (~80 MB)
