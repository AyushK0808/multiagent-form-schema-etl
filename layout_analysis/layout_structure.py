"""
Layout-aware structural modeling using LayoutLMv3.
"""
import torch
import re
from typing import List, Dict, Tuple, Optional
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import logging

logger = logging.getLogger(__name__)

# Label mapping for document structure
LABEL_MAP = {
    0: "paragraph",
    1: "heading",
    2: "list_item",
    3: "table",
    4: "caption",
    5: "other"
}

class LayoutAnalyzer:
    """Analyzes document layout using LayoutLMv3."""
    
    def __init__(self, model_name: str = "microsoft/layoutlmv3-base"):
        logger.info(f"Loading LayoutLMv3 model: {model_name}")
        self.processor = LayoutLMv3Processor.from_pretrained(
            model_name,
            apply_ocr=False
        )
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
        self.model.eval()
    
    def analyze(self, blocks: List[Dict], page_image: Image.Image) -> Dict[str, any]:
        """
        Analyze layout and extract structural information.
        
        Args:
            blocks: List of text blocks with bounding boxes
            page_image: PIL Image of the page
            
        Returns:
            Dictionary with layout predictions and clause graph
        """
        # Prepare inputs
        encoding, word_labels, words = self._prepare_inputs(blocks, page_image)
        
        if encoding is None:
            logger.warning("No valid inputs for layout analysis")
            return {"clause_graph": {}, "predictions": []}
        
        # Run model inference
        predictions = self._predict(encoding)
        
        # Build clause graph from predictions
        clause_graph = self._build_clause_graph(words, predictions, word_labels)
        
        return {
            "clause_graph": clause_graph,
            "predictions": list(zip(words, predictions)),
            "num_clauses": len(clause_graph)
        }
    
    def _prepare_inputs(self, blocks: List[Dict], page_image: Image.Image) -> Tuple:
        """Prepare inputs for LayoutLMv3 with improved word-bbox alignment."""
        words = []
        boxes = []
        word_labels = []  # Track which block each word comes from
        width, height = page_image.size
        
        for block_idx, block in enumerate(blocks):
            if not block.get("bbox"):
                # Handle blocks without bbox (e.g., OCR blocks)
                # Create a dummy bbox spanning the page
                bbox = (0, 0, width, height)
            else:
                bbox = block["bbox"]
            
            # Normalize bbox to 1000x1000 scale (LayoutLMv3 format)
            norm_bbox = self._normalize_bbox(bbox, width, height)
            
            # Tokenize text into words
            text = block["text"]
            block_words = text.split()
            
            for word in block_words:
                if word.strip():
                    words.append(word)
                    boxes.append(norm_bbox)
                    word_labels.append(block_idx)
        
        if not words:
            return None, [], []
        
        # Encode for model
        try:
            encoding = self.processor(
                page_image,
                words,
                boxes=boxes,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return None, [], []
        
        return encoding, word_labels, words
    
    def _normalize_bbox(self, bbox: Tuple[float, float, float, float], 
                       width: int, height: int) -> List[int]:
        """Normalize bounding box to 1000x1000 scale."""
        x0, y0, x1, y1 = bbox
        return [
            int(1000 * x0 / width),
            int(1000 * y0 / height),
            int(1000 * x1 / width),
            int(1000 * y1 / height),
        ]
    
    def _predict(self, encoding: Dict) -> List[int]:
        """Run model inference and return label predictions."""
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Get predicted labels
        predictions = outputs.logits.argmax(-1).squeeze()
        
        # Handle single prediction vs batch
        if predictions.dim() == 0:
            predictions = predictions.unsqueeze(0)
        
        return predictions.tolist()
    
    def _build_clause_graph(self, words: List[str], predictions: List[int], 
                           word_labels: List[int]) -> Dict[str, List[str]]:
        """
        Build hierarchical clause graph from layout predictions.
        
        Args:
            words: List of words
            predictions: List of predicted labels
            word_labels: List tracking which block each word belongs to
            
        Returns:
            Clause graph mapping section IDs to text content
        """
        clause_graph = {}
        current_clause = None
        
        for i, (word, label_id) in enumerate(zip(words, predictions)):
            label = LABEL_MAP.get(label_id, "other")
            
            # Detect clause headings (section numbers like "1.", "1.1", etc.)
            if label == "heading" or self._is_section_number(word):
                # Start new clause
                section_key = self._extract_section_key(word, words[i:i+5])
                if section_key:
                    current_clause = section_key
                    if current_clause not in clause_graph:
                        clause_graph[current_clause] = []
            
            # Add content to current clause
            elif current_clause and label in ["paragraph", "list_item"]:
                clause_graph[current_clause].append(word)
        
        # Post-process: join words into sentences
        for key in clause_graph:
            clause_graph[key] = " ".join(clause_graph[key])
        
        return clause_graph
    
    def _is_section_number(self, word: str) -> bool:
        """Check if word is a section number."""
        # Match patterns like: "1.", "1.1", "Article 5", "Section 3.2"
        patterns = [
            r'^\d+\.(\d+\.?)*$',  # 1., 1.1, 1.1.1
            r'^Article\s+\d+',
            r'^Section\s+\d+',
            r'^\(\d+\)',  # (1), (2)
            r'^[A-Z]\.',  # A., B.
        ]
        return any(re.match(p, word, re.IGNORECASE) for p in patterns)
    
    def _extract_section_key(self, word: str, context: List[str]) -> Optional[str]:
        """Extract clean section key from heading."""
        # Clean common patterns
        word = word.rstrip('.')
        
        # Handle multi-word headings like "Article 5"
        if word.lower() in ["article", "section"]:
            for next_word in context[1:3]:
                if next_word.isdigit():
                    return f"{word}_{next_word}"
        
        # Return numeric sections
        if re.match(r'^\d+(\.\d+)*$', word):
            return word
        
        # Return parenthetical numbers
        match = re.match(r'^\((\d+)\)$', word)
        if match:
            return match.group(1)
        
        return None


def layout_and_structure(blocks: List[Dict], page_image: Image.Image) -> Dict:
    """
    Convenience function for layout analysis.
    
    Args:
        blocks: List of document blocks
        page_image: PIL Image of the page
        
    Returns:
        Clause graph dictionary
    """
    analyzer = LayoutAnalyzer()
    result = analyzer.analyze(blocks, page_image)
    return result["clause_graph"]