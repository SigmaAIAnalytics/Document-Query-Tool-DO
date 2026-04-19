import { useState, useRef } from "react";
import { uploadDocument } from "../api";

export default function UploadPanel({ onUploaded }) {
  const [dragging, setDragging] = useState(false);
  const [files, setFiles] = useState([]);
  const [companyName, setCompanyName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [results, setResults] = useState([]);
  const inputRef = useRef();

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const dropped = Array.from(e.dataTransfer.files).filter((f) =>
      f.name.toLowerCase().endsWith(".pdf")
    );
    setFiles(dropped);
  };

  const handleFileChange = (e) => {
    setFiles(Array.from(e.target.files));
  };

  const handleUpload = async () => {
    if (!files.length) return;
    setUploading(true);
    setResults([]);
    setUploadStatus("");
    const newResults = [];

    for (const file of files) {
      try {
        setUploadStatus(`Submitting ${file.name}…`);
        const data = await uploadDocument(file, companyName, (job) => {
          setUploadStatus(`Parsing ${file.name} — waiting for Landing.ai…`);
        });
        newResults.push({ file: file.name, status: "success", data });
      } catch (err) {
        newResults.push({ file: file.name, status: "error", message: err.message });
      }
    }

    setResults(newResults);
    setUploading(false);
    setUploadStatus("");
    setFiles([]);
    setCompanyName("");
    if (inputRef.current) inputRef.current.value = "";
    onUploaded?.();
  };

  return (
    <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
      <h2 className="text-lg font-semibold text-gray-800 mb-4">Upload Filing</h2>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
          dragging ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-blue-400 hover:bg-gray-50"
        }`}
      >
        <div className="text-4xl mb-2">📄</div>
        <p className="text-gray-600 text-sm">
          {files.length
            ? files.map((f) => f.name).join(", ")
            : "Drag & drop PDFs here, or click to select"}
        </p>
        <p className="text-xs text-gray-400 mt-1">10-K and 8-K filings only (PDF)</p>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          multiple
          className="hidden"
          onChange={handleFileChange}
        />
      </div>

      {/* Company name */}
      <input
        type="text"
        value={companyName}
        onChange={(e) => setCompanyName(e.target.value)}
        placeholder="Company name (optional — auto-detected from filename)"
        className="mt-4 w-full border border-gray-200 rounded-lg px-4 py-2 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-400"
      />

      <button
        onClick={handleUpload}
        disabled={!files.length || uploading}
        className="mt-4 w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-medium py-2 rounded-lg transition-colors text-sm"
      >
        {uploading ? (uploadStatus || "Uploading…") : `Upload ${files.length ? `${files.length} file(s)` : ""}`}
      </button>

      {/* Upload progress bar placeholder while uploading */}
      {uploading && (
        <div className="mt-3 h-1.5 bg-gray-100 rounded-full overflow-hidden">
          <div className="h-full bg-blue-500 rounded-full animate-pulse w-2/3" />
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div className="mt-4 space-y-2">
          {results.map((r, i) => (
            <div
              key={i}
              className={`text-xs px-3 py-2 rounded-lg flex items-start gap-2 ${
                r.status === "success"
                  ? "bg-green-50 text-green-700 border border-green-200"
                  : "bg-red-50 text-red-700 border border-red-200"
              }`}
            >
              <span>{r.status === "success" ? "✓" : "✗"}</span>
              <span>
                <strong>{r.file}</strong>
                {r.status === "success"
                  ? ` — ${r.data.chunk_count} chunks indexed (${r.data.filing_type})`
                  : ` — ${r.message}`}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
