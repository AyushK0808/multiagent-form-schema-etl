# ingestion.py
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from typing import List, Dict

def ingest_pdf(pdf_path: str) -> List[Dict]:
    """
    Ingests a contract PDF and returns a unified representation
    with text and bounding boxes. OCR is applied as fallback.
    """
    doc = fitz.open(pdf_path)
    blocks_out = []

    for page_idx, page in enumerate(doc):
        blocks = page.get_text("blocks")

        text_volume = sum(len(b[4].strip()) for b in blocks) if blocks else 0

        # OCR fallback for scanned / low-text pages
        if text_volume < 50:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img)

            blocks_out.append({
                "page": page_idx,
                "text": ocr_text,
                "bbox": None
            })
        else:
            for b in blocks:
                x0, y0, x1, y1, text = b[:5]
                if text.strip():
                    blocks_out.append({
                        "page": page_idx,
                        "text": text.strip(),
                        "bbox": (x0, y0, x1, y1)
                    })

    return blocks_out
