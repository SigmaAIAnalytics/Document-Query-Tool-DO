const BASE = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? "http://localhost:8000" : "/api");

export async function uploadDocument(file, companyName, onProgress) {
  const form = new FormData();
  form.append("file", file);
  form.append("company_name", companyName || "");

  // Submit — returns immediately with a job_id
  const res = await fetch(`${BASE}/documents/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Upload failed");
  }
  const { job_id } = await res.json();

  // Poll until done
  while (true) {
    await new Promise((r) => setTimeout(r, 5000));
    const statusRes = await fetch(`${BASE}/documents/jobs/${job_id}`);
    if (!statusRes.ok) throw new Error("Failed to check job status");
    const job = await statusRes.json();

    onProgress?.(job);

    if (job.status === "done") return job;
    if (job.status === "error") throw new Error(job.error || "Processing failed");
  }
}

export async function listDocuments() {
  const res = await fetch(`${BASE}/documents`);
  if (!res.ok) throw new Error("Failed to fetch documents");
  return res.json();
}

export async function deleteDocument(docId) {
  const res = await fetch(`${BASE}/documents/${docId}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete document");
  return res.json();
}

// Pages
export function pageImageUrl(docId, page0idx, box, zoom = 2.0) {
  const params = new URLSearchParams({ zoom: String(zoom) });
  if (box && box.left != null) {
    params.set("hl_left", box.left);
    params.set("hl_top", box.top);
    params.set("hl_right", box.right);
    params.set("hl_bottom", box.bottom);
  }
  return `${BASE}/pages/${docId}/page/${page0idx}?${params.toString()}`;
}

export async function getPageCount(docId) {
  const res = await fetch(`${BASE}/pages/${docId}/page-count`);
  if (!res.ok) return null;
  return (await res.json()).page_count;
}

export async function getSections(docId) {
  const res = await fetch(`${BASE}/pages/${docId}/sections`);
  if (!res.ok) return [];
  return (await res.json()).sections || [];
}

export async function getPageChunks(docId, page0idx) {
  const res = await fetch(`${BASE}/pages/${docId}/page/${page0idx}/chunks`);
  if (!res.ok) return [];
  return (await res.json()).chunks || [];
}

// Prompts
export async function listPrompts(docId) {
  const url = docId ? `${BASE}/prompts?doc_id=${docId}` : `${BASE}/prompts`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch prompts");
  return res.json();
}

export async function savePrompt(name, promptText, docId) {
  const res = await fetch(`${BASE}/prompts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, prompt_text: promptText, doc_id: docId || null }),
  });
  if (!res.ok) throw new Error("Failed to save prompt");
  return res.json();
}

export async function deletePrompt(promptId) {
  const res = await fetch(`${BASE}/prompts/${promptId}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete prompt");
  return res.json();
}

export async function* runPrompt(promptId, docId) {
  const res = await fetch(`${BASE}/prompts/${promptId}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id: docId || null }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Run failed");
  }
  yield* _readSSE(res);
}

async function* _readSSE(res) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const raw = line.slice(6).trim();
      if (raw === "[DONE]") return;
      try {
        const parsed = JSON.parse(raw);
        if (parsed.type === "error") throw new Error(parsed.message);
        yield parsed;
      } catch (e) {
        if (e.message && !e.message.includes("JSON")) throw e;
      }
    }
  }
}

export async function* streamQuery(question, docId, anchorChunkIds = [], anchorDocIds = []) {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      doc_id: docId || null,
      anchor_chunk_ids: anchorChunkIds.length ? anchorChunkIds : null,
      anchor_doc_ids: anchorDocIds.length ? anchorDocIds : null,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Query failed");
  }
  yield* _readSSE(res);
}
