import { useState, useEffect } from "react";
import { listPrompts, savePrompt, deletePrompt, runPrompt } from "../api";

export default function SavedPrompts({ selectedDocId, onRunPrompt }) {
  const [prompts, setPrompts] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [name, setName] = useState("");
  const [text, setText] = useState("");
  const [saving, setSaving] = useState(false);
  const [running, setRunning] = useState(null);

  const fetchPrompts = async () => {
    try {
      const data = await listPrompts(selectedDocId);
      setPrompts(data);
    } catch {}
  };

  useEffect(() => { fetchPrompts(); }, [selectedDocId]);

  const handleSave = async () => {
    if (!name.trim() || !text.trim()) return;
    setSaving(true);
    try {
      await savePrompt(name.trim(), text.trim(), selectedDocId);
      setName(""); setText(""); setShowForm(false);
      fetchPrompts();
    } catch (e) {
      alert(e.message);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm("Delete this prompt?")) return;
    try {
      await deletePrompt(id);
      fetchPrompts();
    } catch (e) {
      alert(e.message);
    }
  };

  const handleRun = async (prompt) => {
    setRunning(prompt.id);
    try {
      onRunPrompt?.(prompt, selectedDocId);
    } finally {
      setRunning(null);
    }
  };

  return (
    <div className="bg-white rounded-2xl border border-gray-200 p-4 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-gray-800">Saved Prompts</h2>
        <button
          onClick={() => setShowForm((v) => !v)}
          className="text-xs bg-blue-600 hover:bg-blue-700 text-white px-2.5 py-1 rounded-lg"
        >
          {showForm ? "Cancel" : "+ New"}
        </button>
      </div>

      {showForm && (
        <div className="mb-4 space-y-2 border border-blue-100 bg-blue-50 rounded-xl p-3">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Prompt name (e.g. Check revenue)"
            className="w-full text-xs border border-gray-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-300"
          />
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter your analysis prompt…"
            rows={3}
            className="w-full text-xs border border-gray-200 rounded-lg px-3 py-2 resize-none focus:outline-none focus:ring-2 focus:ring-blue-300"
          />
          {selectedDocId && (
            <p className="text-xs text-blue-600">Will be scoped to the selected document</p>
          )}
          <button
            onClick={handleSave}
            disabled={saving || !name.trim() || !text.trim()}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white text-xs py-1.5 rounded-lg"
          >
            {saving ? "Saving…" : "Save Prompt"}
          </button>
        </div>
      )}

      {prompts.length === 0 ? (
        <p className="text-xs text-gray-400 text-center py-4">
          No saved prompts yet. Create one to re-run analyses quickly.
        </p>
      ) : (
        <ul className="space-y-2">
          {prompts.map((p) => (
            <li key={p.id} className="border border-gray-100 rounded-xl p-3 hover:border-gray-200 transition-colors">
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <p className="text-xs font-semibold text-gray-800 truncate">{p.name}</p>
                  <p className="text-xs text-gray-400 mt-0.5 line-clamp-2">{p.prompt_text}</p>
                </div>
                <div className="flex gap-1 shrink-0 mt-0.5">
                  <button
                    onClick={() => handleRun(p)}
                    disabled={running === p.id}
                    className="text-xs bg-green-600 hover:bg-green-700 disabled:bg-gray-300 text-white px-2 py-0.5 rounded"
                    title="Run this prompt"
                  >
                    {running === p.id ? "…" : "▶ Run"}
                  </button>
                  <button
                    onClick={() => handleDelete(p.id)}
                    className="text-gray-300 hover:text-red-500 text-base leading-none px-1"
                    title="Delete"
                  >
                    ×
                  </button>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
