import { useState, useEffect } from "react";
import { deleteDocument, getSections, listParsedDocuments, reindexDocument } from "../api";

const FILING_COLORS = {
  "10-K": "bg-blue-100 text-blue-700",
  "8-K": "bg-purple-100 text-purple-700",
  Unknown: "bg-gray-100 text-gray-600",
};

export default function DocumentLibrary({ documents, onDeleted, onSelectDoc, selectedDocId, onSectionClick }) {
  const [deleting, setDeleting] = useState(null);
  const [sections, setSections] = useState([]);
  const [sectionsLoading, setSectionsLoading] = useState(false);
  const [showSections, setShowSections] = useState(false);
  const [parsed, setParsed] = useState([]);
  const [restoring, setRestoring] = useState(null);
  const [showRestore, setShowRestore] = useState(false);

  useEffect(() => {
    if (!selectedDocId) { setSections([]); setShowSections(false); return; }
    setSectionsLoading(true);
    getSections(selectedDocId)
      .then((s) => { setSections(s); if (s.length) setShowSections(true); })
      .catch(() => setSections([]))
      .finally(() => setSectionsLoading(false));
  }, [selectedDocId]);

  useEffect(() => {
    listParsedDocuments().then(setParsed).catch(() => setParsed([]));
  }, [documents]);

  const handleDelete = async (doc) => {
    if (!window.confirm(`Delete "${doc.filename}"?`)) return;
    setDeleting(doc.doc_id);
    try { await deleteDocument(doc.doc_id); onDeleted?.(); }
    catch (err) { alert(err.message); }
    finally { setDeleting(null); }
  };

  const handleRestore = async (item) => {
    setRestoring(item.json_filename);
    try {
      await reindexDocument(item.json_filename);
      onDeleted?.(); // reuse to trigger document list refresh
    } catch (err) {
      alert(err.message);
    } finally {
      setRestoring(null);
    }
  };

  return (
    <div className="bg-white rounded-2xl border border-gray-200 p-4 shadow-sm">
      <h2 className="text-sm font-semibold text-gray-800 mb-3">
        Document Library
        {documents.length > 0 && <span className="ml-1 text-xs text-gray-400 font-normal">({documents.length})</span>}
      </h2>

      {documents.length === 0 ? (
        <p className="text-xs text-gray-400 text-center py-4">No filings uploaded yet.</p>
      ) : (
        <ul className="space-y-1.5 max-h-72 overflow-y-auto pr-1">
          {documents.map((doc) => (
            <li key={doc.doc_id}>
              <div
                onClick={() => onSelectDoc?.(doc.doc_id)}
                className={`flex items-center justify-between gap-2 px-3 py-2 rounded-xl border cursor-pointer transition-colors ${
                  selectedDocId === doc.doc_id
                    ? "border-blue-400 bg-blue-50"
                    : "border-gray-100 hover:border-gray-300 hover:bg-gray-50"
                }`}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-base shrink-0">📋</span>
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-gray-800 truncate">{doc.company_name}</p>
                    <p className="text-xs text-gray-400 truncate">{doc.filename}</p>
                  </div>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  <span className={`text-xs font-medium px-1.5 py-0.5 rounded-full ${FILING_COLORS[doc.filing_type] || FILING_COLORS.Unknown}`}>
                    {doc.filing_type}
                  </span>
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDelete(doc); }}
                    disabled={deleting === doc.doc_id}
                    className="text-gray-300 hover:text-red-500 transition-colors disabled:opacity-40 text-lg leading-none"
                    title="Delete"
                  >×</button>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}

      {/* Sections */}
      {selectedDocId && sections.length > 0 && (
        <div className="mt-3 border-t border-gray-100 pt-3">
          <button
            onClick={() => setShowSections((v) => !v)}
            className="text-xs font-semibold text-gray-600 flex items-center gap-1 w-full text-left"
          >
            <span>{showSections ? "▾" : "▸"}</span>
            <span>Sections {sectionsLoading ? "…" : `(${sections.length})`}</span>
          </button>
          {showSections && (
            <ul className="mt-2 space-y-0.5 max-h-52 overflow-y-auto">
              {sections.map((s, i) => (
                <li key={i}>
                  <button
                    onClick={() => onSectionClick?.(s)}
                    className="w-full text-left text-xs px-2 py-1 rounded-lg hover:bg-blue-50 hover:text-blue-700 text-gray-600 truncate transition-colors"
                    title={s.text}
                  >
                    <span className="text-gray-300 mr-1">p.{s.page + 1}</span>
                    {s.text}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Restore section */}
      {parsed.length > 0 && (
        <div className="mt-3 border-t border-gray-100 pt-3">
          <button
            onClick={() => setShowRestore((v) => !v)}
            className="text-xs font-semibold text-amber-600 flex items-center gap-1 w-full text-left"
          >
            <span>{showRestore ? "▾" : "▸"}</span>
            <span>Restore removed documents ({parsed.length})</span>
          </button>
          {showRestore && (
            <ul className="mt-2 space-y-1.5 max-h-52 overflow-y-auto pr-1">
              {parsed.map((item) => (
                <li key={item.json_filename} className="flex items-center justify-between gap-2 px-3 py-2 rounded-xl border border-amber-100 bg-amber-50">
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-gray-700 truncate">{item.filename}</p>
                    <p className="text-xs text-gray-400">{item.filing_type} · {item.has_pdf ? "PDF available" : "no PDF"}</p>
                  </div>
                  <button
                    onClick={() => handleRestore(item)}
                    disabled={restoring === item.json_filename}
                    className="shrink-0 text-xs bg-amber-500 hover:bg-amber-600 disabled:bg-gray-300 text-white px-2 py-1 rounded-lg transition-colors"
                  >
                    {restoring === item.json_filename ? "Restoring…" : "Restore"}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
