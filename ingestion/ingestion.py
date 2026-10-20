"""
PDF ingestion with OCR fallback and enhanced error handling.
"""
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from typing import List, Dict, Optional
import logging
from config.config import get_config

logger = logging.getLogger(__name__)

class DocumentBlock:
    """Represents a text block with metadata."""
    
    def __init__(self, page: int, text: str, bbox: Optional[tuple] = None, 
                 confidence: float = 1.0, source: str = "native"):
        self.page = page
        self.text = text.strip()
        self.bbox = bbox  # (x0, y0, x1, y1)
        self.confidence = confidence
        self.source = source  # "native" or "ocr"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "page": self.page,
            "text": self.text,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "source": self.source
        }
    
    def __repr__(self):
        return f"Block(page={self.page}, text={self.text[:50]}..., source={self.source})"


class PDFIngester:
    """Handles PDF ingestion with OCR fallback."""
    
    def __init__(self, ocr_threshold: int = None):
        config = get_config()
        self.ocr_threshold = ocr_threshold or config.processing.ocr_threshold
        self.max_page_size = config.processing.max_page_size
    
    def ingest(self, pdf_path: str) -> tuple[List[DocumentBlock], Dict]:
        """
        Ingest a PDF and return structured blocks with metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (blocks, metadata)
        """
        logger.info(f"Ingesting PDF: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise
        
        blocks = []
        metadata = {
            "total_pages": len(doc),
            "ocr_pages": [],
            "native_pages": [],
            "ocr_mean_confidence": None,
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", "")
        }
        
        for page_idx, page in enumerate(doc):
            page_blocks = self._process_page(page, page_idx, metadata)
            blocks.extend(page_blocks)
        
        
        ocr_conf = [b.confidence for b in blocks if b.source == "ocr"]
        if ocr_conf:
            metadata["ocr_mean_confidence"] = round(sum(ocr_conf) / len(ocr_conf), 4)

        logger.info(f"Ingested {len(blocks)} blocks from {len(doc)} pages")
        doc.close()
        return blocks, metadata
    
    def _process_page(self, page, page_idx: int, metadata: Dict) -> List[DocumentBlock]:
        """Process a single page with OCR fallback."""
        blocks = []
        
        # Try native text extraction first
        text_blocks = page.get_text("blocks")
        text_volume = sum(len(b[4].strip()) for b in text_blocks) if text_blocks else 0
        
        if text_volume < self.ocr_threshold:
            # Use OCR for low-text pages
            logger.debug(f"Page {page_idx}: Using OCR (text_volume={text_volume})")
            blocks.extend(self._ocr_page(page, page_idx))
            metadata["ocr_pages"].append(page_idx)
        else:
            # Use native extraction
            logger.debug(f"Page {page_idx}: Using native extraction (text_volume={text_volume})")
            for b in text_blocks:
                x0, y0, x1, y1, text = b[:5]
                if text.strip():
                    block = DocumentBlock(
                        page=page_idx,
                        text=text.strip(),
                        bbox=(x0, y0, x1, y1),
                        confidence=1.0,
                        source="native"
                    )
                    blocks.append(block)
            metadata["native_pages"].append(page_idx)
        
        return blocks
    
    def _ocr_page(self, page, page_idx: int) -> List[DocumentBlock]:
        """
        Apply OCR to a page and return one block per Tesseract text line.

        Each block's confidence is the mean Tesseract word confidence of its
        words (rescaled to 0-1); its bbox is in page coordinates (points), the
        same space as native blocks.
        """
        try:
            # Render page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution for better OCR

            # Resize if too large
            if max(pix.width, pix.height) > self.max_page_size:
                scale = self.max_page_size / max(pix.width, pix.height)
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            sx = page.rect.width / pix.width
            sy = page.rect.height / pix.height

            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

            lines: Dict[tuple, List[int]] = {}
            for i, word in enumerate(data["text"]):
                try:
                    conf = float(data["conf"][i])
                except (TypeError, ValueError):
                    continue
                if not str(word).strip() or conf < 0:  # conf -1 marks non-word rows
                    continue
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                lines.setdefault(key, []).append(i)

            blocks = []
            for key in sorted(lines):
                idx = lines[key]
                x0 = min(data["left"][i] for i in idx)
                y0 = min(data["top"][i] for i in idx)
                x1 = max(data["left"][i] + data["width"][i] for i in idx)
                y1 = max(data["top"][i] + data["height"][i] for i in idx)
                mean_conf = sum(float(data["conf"][i]) for i in idx) / len(idx) / 100.0
                blocks.append(DocumentBlock(
                    page=page_idx,
                    text=" ".join(str(data["text"][i]).strip() for i in idx),
                    bbox=(x0 * sx, y0 * sy, x1 * sx, y1 * sy),
                    confidence=round(mean_conf, 4),
                    source="ocr",
                ))
            return blocks
        except Exception as e:
            logger.error(f"OCR failed for page {page_idx}: {e}")

        return []


def ingest_pdf(pdf_path: str) -> tuple[List[Dict], Dict]:
    """
    Convenience function for PDF ingestion.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Tuple of (block_dicts, metadata)
    """
    ingester = PDFIngester()
    blocks, metadata = ingester.ingest(pdf_path)
    return [b.to_dict() for b in blocks], metadata