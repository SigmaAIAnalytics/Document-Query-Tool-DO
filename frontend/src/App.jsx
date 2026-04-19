import { useState, useEffect, useCallback } from "react";
import UploadPanel from "./components/UploadPanel";
import DocumentLibrary from "./components/DocumentLibrary";
import ChatInterface from "./components/ChatInterface";
import PDFViewer from "./components/PDFViewer";
import SavedPrompts from "./components/SavedPrompts";
import { listDocuments } from "./api";

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState(null);

  // PDF viewer state
  const [pdfOpen, setPdfOpen] = useState(false);
  const [pdfDocId, setPdfDocId] = useState(null);
  const [pdfPage, setPdfPage] = useState(0);
  const [pdfBox, setPdfBox] = useState(null);

  // Selected chunks for anchored queries (multi-select)
  const [selectedChunks, setSelectedChunks] = useState([]);

  // External prompt injected from Saved Prompts → ChatInterface
  const [externalPrompt, setExternalPrompt] = useState(null);

  // Prompt list refresh trigger
  const [promptsKey, setPromptsKey] = useState(0);

  // Main tab
  const [activeTab, setActiveTab] = useState("chat");

  const fetchDocs = async () => {
    try { setDocuments(await listDocuments()); } catch {}
  };

  useEffect(() => { fetchDocs(); }, []);

  const handleCitationClick = useCallback((citation) => {
    setPdfDocId(citation.doc_id || selectedDocId);
    setPdfPage(citation.page_0idx ?? Math.max(0, (citation.page || 1) - 1));
    setPdfBox(
      citation.box_left != null
        ? { left: citation.box_left, top: citation.box_top, right: citation.box_right, bottom: citation.box_bottom }
        : null
    );
    setPdfOpen(true);
    setActiveTab("pdf");
  }, [selectedDocId]);

  const handleChunkSelect = useCallback((chunk) => {
    setSelectedChunks((prev) => {
      const exists = prev.some((c) => c.chunk_id === chunk.chunk_id);
      return exists ? prev.filter((c) => c.chunk_id !== chunk.chunk_id) : [...prev, chunk];
    });
    // Stay on PDF so user can select more chunks before switching to chat
  }, []);

  const handleRunPrompt = useCallback((prompt, docId) => {
    setExternalPrompt({ text: prompt.prompt_text, docId: docId || prompt.doc_id, label: prompt.name, _ts: Date.now() });
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 font-sans flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-20 shrink-0">
        <div className="px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="SigmaAI Logo" className="h-8 w-auto" />
            <h1 className="text-base font-bold text-gray-900 leading-none">Document Query Tool</h1>
          </div>
          <span className="text-xs text-gray-400 hidden sm:block">Landing.ai · ChromaDB · Claude</span>
        </div>
      </header>

      {/* Three-panel layout */}
      <div className="flex flex-1 overflow-hidden">

        {/* Left sidebar */}
        <div className="w-72 shrink-0 border-r border-gray-200 bg-white overflow-y-auto flex flex-col gap-4 p-4">
          <UploadPanel onUploaded={fetchDocs} />
          <DocumentLibrary
            documents={documents}
            onDeleted={fetchDocs}
            onSelectDoc={(id) => {
              setSelectedDocId(id === selectedDocId ? null : id);
              if (id) { setPdfDocId(id); setPdfPage(0); setPdfBox(null); setPdfOpen(true); }
            }}
            selectedDocId={selectedDocId}
            onSectionClick={(section) => {
              setPdfDocId(selectedDocId);
              setPdfPage(section.page);
              setPdfBox(section.box?.left != null ? section.box : null);
              setPdfOpen(true);
            }}
          />
          <SavedPrompts
            key={promptsKey}
            selectedDocId={selectedDocId}
            onRunPrompt={handleRunPrompt}
          />
        </div>

        {/* Main tabbed area */}
        <div className="flex-1 overflow-hidden flex flex-col">
          {/* Tab bar */}
          <div className="flex items-center gap-1 px-4 pt-3 pb-0 bg-white border-b border-gray-200 shrink-0">
            <button
              onClick={() => setActiveTab("chat")}
              className={`px-4 py-2 text-sm font-medium rounded-t-lg border-b-2 transition-colors ${
                activeTab === "chat"
                  ? "border-blue-600 text-blue-600 bg-white"
                  : "border-transparent text-gray-500 hover:text-gray-700"
              }`}
            >
              💬 Chat
            </button>
            <button
              onClick={() => setActiveTab("pdf")}
              className={`px-4 py-2 text-sm font-medium rounded-t-lg border-b-2 transition-colors flex items-center gap-2 ${
                activeTab === "pdf"
                  ? "border-blue-600 text-blue-600 bg-white"
                  : "border-transparent text-gray-500 hover:text-gray-700"
              }`}
            >
              📄 PDF Viewer
              {pdfOpen && <span className="w-2 h-2 rounded-full bg-blue-500 inline-block" />}
            </button>
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-hidden">
            <div className={`h-full p-4 ${activeTab === "chat" ? "block" : "hidden"}`}>
              <ChatInterface
                selectedDocId={selectedDocId}
                documents={documents}
                onCitationClick={handleCitationClick}
                externalPrompt={externalPrompt}
                onPromptsUpdated={() => setPromptsKey((k) => k + 1)}
                anchorChunks={selectedChunks}
                onClearAnchor={(chunkId) =>
                  chunkId
                    ? setSelectedChunks((prev) => prev.filter((c) => c.chunk_id !== chunkId))
                    : setSelectedChunks([])
                }
              />
            </div>
            <div className={`h-full ${activeTab === "pdf" ? "block" : "hidden"}`}>
              <PDFViewer
                docId={pdfDocId}
                page0idx={pdfPage}
                box={pdfBox}
                selectedChunkIds={selectedChunks.map((c) => c.chunk_id)}
                onChunkSelect={handleChunkSelect}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
