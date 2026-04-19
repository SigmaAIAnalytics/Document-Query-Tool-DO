import sqlite3
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.getenv("PROMPTS_DB_PATH", "./prompts.db"))


def _conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                doc_id      TEXT,
                created_at  TEXT NOT NULL
            )
        """)
        conn.commit()


def save_prompt(name: str, prompt_text: str, doc_id: str = None) -> dict:
    pid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO prompts (id, name, prompt_text, doc_id, created_at) VALUES (?,?,?,?,?)",
            (pid, name, prompt_text, doc_id, ts),
        )
        conn.commit()
    return {"id": pid, "name": name, "prompt_text": prompt_text, "doc_id": doc_id, "created_at": ts}


def list_prompts(doc_id: str = None) -> list[dict]:
    with _conn() as conn:
        if doc_id:
            rows = conn.execute(
                "SELECT * FROM prompts WHERE doc_id=? OR doc_id IS NULL ORDER BY created_at DESC",
                (doc_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM prompts ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_prompt(pid: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM prompts WHERE id=?", (pid,)).fetchone()
    return dict(row) if row else None


def delete_prompt(pid: str) -> bool:
    with _conn() as conn:
        r = conn.execute("DELETE FROM prompts WHERE id=?", (pid,))
        conn.commit()
    return r.rowcount > 0


def update_prompt(pid: str, name: str, prompt_text: str) -> dict | None:
    with _conn() as conn:
        conn.execute(
            "UPDATE prompts SET name=?, prompt_text=? WHERE id=?",
            (name, prompt_text, pid),
        )
        conn.commit()
    return get_prompt(pid)
