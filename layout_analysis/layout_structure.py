import torch
import re
from typing import List, Dict
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# --- Model Initialization ---
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False
)

model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base"
)
model.eval()

# --- Helper Functions (Moved from bbox_utils.py) ---
LABEL_MAP = {
    0: "paragraph",
    1: "heading",
    2: "table",
    3: "other"
}

def normalize_bbox(bbox, width, height):
    x0, y0, x1, y1 = bbox
    return [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height),
    ]

def prepare_layout_inputs(blocks: List[Dict], page_image):
    words, boxes = [], []
    width, height = page_image.size

    for b in blocks:
        # bbox is (x0, y0, x1, y1)
        if not b.get("bbox"):
            continue
            
        # Split text into words to match tokenization needs roughly
        # Note: For production, alignment between words and bboxes needs to be more precise
        curr_text = b["text"]
        curr_box = normalize_bbox(b["bbox"], width, height)
        
        # Simple splitting by whitespace
        for token in curr_text.split():
            words.append(token)
            boxes.append(curr_box)

    if not words: 
        return None, []

    encoding = processor(
        page_image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    return encoding, words

def build_clause_graph(words, labels):
    """
    Deterministic clause hierarchy over LayoutLMv3 predictions.
    """
    clause_graph = {}
    current_clause = None
    
    # Safely handle cases where words/labels length might mismatch due to tokenization
    # In a real scenario, you map tokens back to words. Here we zip 1:1.
    for word, lid in zip(words, labels):
        label = LABEL_MAP.get(lid, "other")

        # Heuristic: If it looks like a section number (1., 1.1), it's a new clause key
        if label == "heading" and re.match(r"^\d+(\.\d+)*", word):
            current_clause = word
            clause_graph[current_clause] = []
        elif current_clause:
            clause_graph[current_clause].append(word)

    return clause_graph

# --- Main Exported Function ---
def layout_and_structure(blocks, page_image):
    encoding, words = prepare_layout_inputs(blocks, page_image)
    
    if encoding is None:
        return {}

    with torch.no_grad():
        outputs = model(**encoding)

    # Get predictions
    labels = outputs.logits.argmax(-1).squeeze().tolist()
    
    return build_clause_graph(words, labels)