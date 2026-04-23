import { useState, useRef, useEffect } from "react";
import { streamQuery, savePrompt } from "../api";
import ReactMarkdown from "react-markdown";

function CitationBadge({ citation, onCitationClick }) {
  return (
    <button
      onClick={() => onCitationClick?.(citation)}
      className="inline-flex items-center gap-1 text-xs bg-blue-50 border border-blue-200 text-blue-700 rounded px-2 py-0.5 mr-1 mb-1 hover:bg-blue-100 transition-colors cursor-pointer"
      title="Click to view source page"
    >
      <span className="font-medium">{citation.filing_type}</span>
      <span className="text-blue-400">·</span>
      <span className="truncate max-w-[140px]">{citation.company_name || citation.filename}</span>
      <span className="text-blue-400">·</span>
      <span>p.{citation.page}</span>
      <span className="text-blue-300 ml-0.5">↗</span>
    </button>
  );
}

function Message({ msg, onCitationClick }) {
  return (
    <div className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          msg.role === "user"
            ? "bg-blue-600 text-white rounded-br-sm"
            : "bg-white border border-gray-200 text-gray-800 rounded-bl-sm shadow-sm"
        }`}
      >
        {msg.label && (
          <p className="text-xs font-semibold text-blue-600 mb-1.5 flex items-center gap-1">
            ▶ {msg.label}
          </p>
        )}
        {msg.status && (
          <p className="text-xs text-blue-500 italic mb-1">{msg.status}</p>
        )}
        {msg.role === "user" ? (
          <p className="whitespace-pre-wrap">{msg.content}</p>
        ) : (
          <ReactMarkdown
            components={{
              p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
              strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
              ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>,
              ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>,
              li: ({ children }) => <li className="text-sm">{children}</li>,
              h1: ({ children }) => <h1 className="font-bold text-base mb-2 mt-3">{children}</h1>,
              h2: ({ children }) => <h2 className="font-bold text-sm mb-1.5 mt-2">{children}</h2>,
              h3: ({ children }) => <h3 className="font-semibold text-sm mb-1 mt-2">{children}</h3>,
              code: ({ inline, children }) =>
                inline
                  ? <code className="bg-gray-100 rounded px-1 py-0.5 text-xs font-mono">{children}</code>
                  : <pre className="bg-gray-100 rounded p-2 text-xs font-mono overflow-x-auto mb-2"><code>{children}</code></pre>,
              table: ({ children }) => (
                <div className="overflow-x-auto mb-2">
                  <table className="text-xs border-collapse w-full">{children}</table>
                </div>
              ),
              th: ({ children }) => <th className="border border-gray-300 px-2 py-1 bg-gray-50 font-semibold text-left">{children}</th>,
              td: ({ children }) => <td className="border border-gray-300 px-2 py-1">{children}</td>,
            }}
          >
            {msg.content}
          </ReactMarkdown>
        )}
        {msg.citations?.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-100">
            <p className="text-xs text-gray-400 mb-1.5 font-medium uppercase tracking-wide">Sources — click to view page</p>
            <div className="flex flex-wrap">
              {msg.citations.map((c, i) => (
                <CitationBadge key={i} citation={c} onCitationClick={onCitationClick} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function SavePromptDialog({ question, docId, onSaved, onClose }) {
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    if (!name.trim()) return;
    setSaving(true);
    try {
      await savePrompt(name.trim(), question, docId);
      onSaved?.();
      onClose();
    } catch (e) {
      alert(e.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="absolute bottom-20 left-0 right-0 mx-6 bg-white border border-blue-200 rounded-xl shadow-lg p-4 z-10">
      <p className="text-xs font-semibold text-gray-700 mb-2">Save prompt as:</p>
      <input
        autoFocus
        value={name}
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && handleSave()}
        placeholder="e.g. Check revenue figures"
        className="w-full text-xs border border-gray-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-300 mb-2"
      />
      <div className="flex gap-2">
        <button onClick={handleSave} disabled={saving || !name.trim()} className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white text-xs py-1.5 rounded-lg">
          {saving ? "Saving…" : "Save"}
        </button>
        <button onClick={onClose} className="flex-1 border border-gray-200 text-gray-600 text-xs py-1.5 rounded-lg hover:bg-gray-50">Cancel</button>
      </div>
    </div>
  );
}

export default function ChatInterface({ selectedDocId, documents, onCitationClick, externalPrompt, onPromptsUpdated, anchorChunks = [], onClearAnchor }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [saveDialog, setSaveDialog] = useState(null);
  const bottomRef = useRef();
  const inputRef = useRef();

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  // Handle externally injected prompts (from Saved Prompts panel)
  useEffect(() => {
    if (externalPrompt) runQuestion(externalPrompt.text, externalPrompt.docId, externalPrompt.label);
  }, [externalPrompt]);

  const selectedDoc = documents?.find((d) => d.doc_id === selectedDocId);

  const runQuestion = async (question, docId, label) => {
    if (!question || loading) return;
    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    const assistantId = Date.now();
    setMessages((prev) => [...prev, { id: assistantId, role: "assistant", content: "", citations: [], label }]);
    setLoading(true);

    try {
      const anchorChunkIds = anchorChunks.map((c) => c.chunk_id);
      const anchorDocIds = anchorChunks.map((c) => c.doc_id).filter(Boolean);
      for await (const chunk of streamQuery(question, docId ?? selectedDocId, anchorChunkIds, anchorDocIds)) {
        if (chunk.type === "citations") {
          setMessages((prev) => prev.map((m) => m.id === assistantId ? { ...m, citations: chunk.citations } : m));
        } else if (chunk.type === "text") {
          setMessages((prev) => prev.map((m) => m.id === assistantId ? { ...m, content: m.content + chunk.text, status: null } : m));
        } else if (chunk.type === "status") {
          setMessages((prev) => prev.map((m) => m.id === assistantId ? { ...m, status: chunk.message } : m));
        }
      }
    } catch (err) {
      setError(err.message);
      setMessages((prev) => prev.filter((m) => m.id !== assistantId));
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleSend = () => {
    const question = input.trim();
    if (!question || loading) return;
    setInput("");
    runQuestion(question);
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-sm flex flex-col h-full relative">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between shrink-0">
        <h2 className="text-lg font-semibold text-gray-800">Ask the Filings</h2>
        {selectedDoc && (
          <span className="text-xs bg-blue-50 border border-blue-200 text-blue-700 px-2 py-1 rounded-lg truncate max-w-[200px]">
            Scoped: {selectedDoc.company_name || selectedDoc.filename}
          </span>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-sm text-gray-400 mt-10">
            <p className="text-3xl mb-3">💬</p>
            <p>Ask a question about your SEC filings.</p>
            <p className="mt-1 text-xs">e.g. "What were the key risk factors in the 10-K?"</p>
            <p className="mt-0.5 text-xs">Citations are clickable — they open the source page in the PDF viewer.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <Message key={msg.id || i} msg={msg} onCitationClick={onCitationClick} />
        ))}
        {loading && messages[messages.length - 1]?.content === "" && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm">
              <span className="inline-flex gap-1">
                <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce [animation-delay:0ms]" />
                <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce [animation-delay:150ms]" />
                <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce [animation-delay:300ms]" />
              </span>
            </div>
          </div>
        )}
        {error && <div className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-xl px-4 py-3">{error}</div>}
        <div ref={bottomRef} />
      </div>

      {/* Save prompt dialog */}
      {saveDialog && (
        <SavePromptDialog
          question={saveDialog}
          docId={selectedDocId}
          onSaved={() => onPromptsUpdated?.()}
          onClose={() => setSaveDialog(null)}
        />
      )}

      {/* Anchor chunks banner */}
      {anchorChunks.length > 0 && (
        <div className="mx-6 mb-0 mt-2 bg-blue-50 border border-blue-200 rounded-xl px-3 py-2 text-xs space-y-1">
          <div className="flex items-center justify-between mb-1">
            <span className="text-blue-600 font-semibold">
              {anchorChunks.length} anchor{anchorChunks.length > 1 ? "s" : ""} selected
            </span>
            <button
              onClick={() => onClearAnchor()}
              className="text-blue-400 hover:text-blue-700 font-bold text-xs"
              title="Clear all"
            >Clear all</button>
          </div>
          {anchorChunks.map((c) => (
            <div key={c.chunk_id} className="flex items-start gap-2">
              <span className="text-blue-400 shrink-0">{c.chunk_type === "table" ? "📊" : "📝"}</span>
              <span className="text-blue-700 flex-1 truncate">
                p.{c.page ?? "?"} — {c.text?.slice(0, 80)}…
              </span>
              <button
                onClick={() => onClearAnchor(c.chunk_id)}
                className="text-blue-400 hover:text-blue-700 shrink-0 font-bold"
                title="Remove"
              >×</button>
            </div>
          ))}
        </div>
      )}

      {/* Input */}
      <div className="px-6 py-4 border-t border-gray-100 shrink-0">
        <div className="flex gap-2 items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            rows={1}
            placeholder="Ask a question… (Enter to send)"
            className="flex-1 resize-none border border-gray-200 rounded-xl px-4 py-3 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-400 max-h-32 overflow-y-auto"
            disabled={loading}
          />
          <button
            onClick={() => input.trim() && setSaveDialog(input.trim())}
            disabled={!input.trim()}
            className="shrink-0 border border-gray-200 hover:border-blue-300 text-gray-500 hover:text-blue-600 rounded-xl px-3 py-3 text-xs transition-colors disabled:opacity-40"
            title="Save this prompt"
          >
            💾
          </button>
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="shrink-0 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white rounded-xl px-5 py-3 text-sm font-medium transition-colors"
          >
            {loading ? "…" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
