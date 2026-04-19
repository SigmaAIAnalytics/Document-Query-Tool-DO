from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import documents, query, pages, prompts

app = FastAPI(title="SEC Filings RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_origin_regex=r"https://(.*\.vercel\.app|.*\.ondigitalocean\.app)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(pages.router, prefix="/pages", tags=["pages"])
app.include_router(prompts.router, prefix="/prompts", tags=["prompts"])
app.include_router(query.router, tags=["query"])


@app.get("/health")
def health():
    return {"status": "ok"}
