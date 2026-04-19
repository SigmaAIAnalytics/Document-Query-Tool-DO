import { useState, useEffect, useRef } from "react";
import { pageImageUrl, getPageCount, getPageChunks } from "../api";

const ZOOM_LEVELS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0];

export default function PDFViewer({ docId, page0idx, box, selectedChunkIds = [], onChunkSelect }) {
  const selectedSet = new Set(selectedChunkIds);
  const [currentPage, setCurrentPage] = useState(page0idx ?? 0);
  const [pageCount, setPageCount] = useState(null);
  const [imgError, setImgError] = useState(false);
  const [zoomIdx, setZoomIdx] = useState(2); // default 1.0
  const [jumpValue, setJumpValue] = useState("");
  const [chunks, setChunks] = useState([]);
  const [hoveredChunkId, setHoveredChunkId] = useState(null);
  const [showOverlays, setShowOverlays] = useState(true);
  const jumpRef = useRef();

  const zoom = ZOOM_LEVELS[zoomIdx];

  useEffect(() => {
    setCurrentPage(page0idx ?? 0);
    setImgError(false);
  }, [page0idx, docId]);

  useEffect(() => {
    if (!docId) return;
    getPageCount(docId).then((n) => setPageCount(n));
  }, [docId]);

  useEffect(() => {
    if (!docId) { setChunks([]); return; }
    getPageChunks(docId, currentPage).then(setChunks);
  }, [docId, currentPage]);

  if (!docId) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-400 gap-3">
        <span className="text-5xl">📄</span>
        <p className="text-sm">Click a citation in the chat to view the source page</p>
        <p className="text-xs">or navigate sections from the document library</p>
      </div>
    );
  }

  const imgUrl = pageImageUrl(docId, currentPage, box, zoom * 2);
  const hasPrev = currentPage > 0;
  const hasNext = pageCount != null && currentPage < pageCount - 1;

  const goTo = (n) => {
    if (!isNaN(n) && n >= 1 && (pageCount == null || n <= pageCount)) {
      setCurrentPage(n - 1);
      setImgError(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-100">
      {/* Toolbar */}
      <div className="flex items-center gap-3 px-4 py-2 bg-white border-b border-gray-200 shrink-0 flex-wrap">

        {/* Page navigation */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => { setCurrentPage((p) => Math.max(0, p - 1)); setImgError(false); }}
            disabled={!hasPrev}
            className="w-7 h-7 flex items-center justify-center rounded border border-gray-200 text-gray-600 disabled:opacity-30 hover:bg-gray-50 text-sm"
          >‹</button>

          <div className="flex items-center gap-1 text-xs text-gray-600">
            <input
              ref={jumpRef}
              type="number"
              value={jumpValue}
              min={1}
              max={pageCount || 9999}
              placeholder={String(currentPage + 1)}
              onChange={(e) => setJumpValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") { goTo(parseInt(jumpValue, 10)); setJumpValue(""); }
              }}
              onBlur={() => { if (jumpValue) { goTo(parseInt(jumpValue, 10)); setJumpValue(""); } }}
              className="w-12 text-center border border-gray-200 rounded px-1 py-0.5 focus:outline-none focus:ring-2 focus:ring-blue-300"
            />
            <span className="text-gray-400">/ {pageCount ?? "…"}</span>
          </div>

          <button
            onClick={() => { setCurrentPage((p) => p + 1); setImgError(false); }}
            disabled={!hasNext}
            className="w-7 h-7 flex items-center justify-center rounded border border-gray-200 text-gray-600 disabled:opacity-30 hover:bg-gray-50 text-sm"
          >›</button>
        </div>

        <div className="w-px h-5 bg-gray-200" />

        {/* Zoom controls */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setZoomIdx((i) => Math.max(0, i - 1))}
            disabled={zoomIdx === 0}
            className="w-7 h-7 flex items-center justify-center rounded border border-gray-200 text-gray-600 disabled:opacity-30 hover:bg-gray-50 font-bold"
          >−</button>
          <span className="text-xs text-gray-600 w-10 text-center">{Math.round(zoom * 100)}%</span>
          <button
            onClick={() => setZoomIdx((i) => Math.min(ZOOM_LEVELS.length - 1, i + 1))}
            disabled={zoomIdx === ZOOM_LEVELS.length - 1}
            className="w-7 h-7 flex items-center justify-center rounded border border-gray-200 text-gray-600 disabled:opacity-30 hover:bg-gray-50 font-bold"
          >+</button>
          <button
            onClick={() => setZoomIdx(2)}
            className="text-xs text-gray-400 hover:text-blue-600 px-1"
            title="Reset zoom"
          >Reset</button>
        </div>

        <div className="w-px h-5 bg-gray-200" />

        {/* Overlay toggle */}
        <button
          onClick={() => setShowOverlays((v) => !v)}
          className={`text-xs px-2 py-1 rounded border transition-colors ${
            showOverlays
              ? "border-blue-300 text-blue-600 bg-blue-50"
              : "border-gray-200 text-gray-400 hover:text-gray-600"
          }`}
          title="Toggle selectable chunk overlays"
        >
          {showOverlays ? "Overlays on" : "Overlays off"}
        </button>

        {selectedChunkIds.length > 0 && (
          <span className="text-xs text-blue-600 bg-blue-50 border border-blue-200 rounded px-2 py-1">
            {selectedChunkIds.length} chunk{selectedChunkIds.length > 1 ? "s" : ""} selected
          </span>
        )}
      </div>

      {/* Page image with chunk overlays */}
      <div className="flex-1 overflow-auto p-4">
        {imgError ? (
          <div className="flex flex-col items-center justify-center h-full text-sm text-gray-400 gap-2">
            <span className="text-3xl">⚠️</span>
            <p>Page image unavailable — PDF may not be stored on the server.</p>
          </div>
        ) : (
          <div className="flex justify-center">
            <div style={{ position: "relative", display: "inline-block", width: `${zoom * 100}%` }}>
              <img
                key={`${docId}-${currentPage}-${JSON.stringify(box)}-${zoom}`}
                src={imgUrl}
                alt={`Page ${currentPage + 1}`}
                style={{ width: "100%", display: "block" }}
                className="shadow-lg rounded bg-white"
                onError={() => setImgError(true)}
              />

              {/* Chunk overlays */}
              {showOverlays && chunks.map((chunk) => {
                const isSelected = selectedSet.has(chunk.chunk_id);
                const isHovered = chunk.chunk_id === hoveredChunkId;
                const isTable = chunk.chunk_type === "table";
                return (
                  <div
                    key={chunk.chunk_id}
                    title={chunk.text.slice(0, 120)}
                    onClick={() => onChunkSelect?.(chunk)}
                    onMouseEnter={() => setHoveredChunkId(chunk.chunk_id)}
                    onMouseLeave={() => setHoveredChunkId(null)}
                    style={{
                      position: "absolute",
                      left: `${chunk.box_left * 100}%`,
                      top: `${chunk.box_top * 100}%`,
                      width: `${(chunk.box_right - chunk.box_left) * 100}%`,
                      height: `${(chunk.box_bottom - chunk.box_top) * 100}%`,
                      cursor: "pointer",
                      boxSizing: "border-box",
                      border: isSelected
                        ? "2px solid #2563eb"
                        : isHovered
                        ? `1px solid ${isTable ? "#d97706" : "#93c5fd"}`
                        : "1px solid transparent",
                      backgroundColor: isSelected
                        ? "rgba(37,99,235,0.15)"
                        : isHovered
                        ? isTable ? "rgba(217,119,6,0.08)" : "rgba(147,197,253,0.12)"
                        : "transparent",
                      borderRadius: "2px",
                      transition: "background-color 0.1s, border-color 0.1s",
                    }}
                  />
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
