import os
import fitz  # PyMuPDF
from pathlib import Path

PDF_DIR = Path(os.getenv("PDF_STORAGE_DIR", "./uploaded_pdfs"))


def save_pdf(doc_id: str, file_bytes: bytes):
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    (PDF_DIR / f"{doc_id}.pdf").write_bytes(file_bytes)


def render_page(doc_id: str, page_num: int, highlight_box: dict = None, zoom: float = 2.0) -> bytes:
    pdf_path = PDF_DIR / f"{doc_id}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found for doc_id={doc_id}")

    doc = fitz.open(str(pdf_path))
    if page_num < 0 or page_num >= len(doc):
        raise ValueError(f"Page {page_num} out of range (0-{len(doc)-1})")

    page = doc[page_num]

    if highlight_box:
        rect = page.rect
        w, h = rect.width, rect.height
        hl = fitz.Rect(
            highlight_box["left"] * w,
            highlight_box["top"] * h,
            highlight_box["right"] * w,
            highlight_box["bottom"] * h,
        )
        shape = page.new_shape()
        shape.draw_rect(hl)
        shape.finish(color=(1, 0.7, 0), fill=(1, 0.9, 0.2), fill_opacity=0.4, width=2)
        shape.commit()

    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    doc.close()
    return pix.tobytes("png")


def get_page_count(doc_id: str) -> int:
    pdf_path = PDF_DIR / f"{doc_id}.pdf"
    if not pdf_path.exists():
        return 0
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count
