# Standard library
import re
from typing import List, Dict

# PyTorch
import torch

# Hugging Face Transformers (LayoutLMv3)
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification
)

# Image handling
from PIL import Image


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
        if not b["bbox"]:
            continue
        for token in b["text"].split():
            words.append(token)
            boxes.append(normalize_bbox(b["bbox"], width, height))

    encoding = processor(
        page_image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    return encoding, words

LABEL_MAP = {
    0: "paragraph",
    1: "heading",
    2: "table",
    3: "other"
}

def build_clause_graph(words, labels):
    """
    Deterministic clause hierarchy over LayoutLMv3 predictions.
    """
    clause_graph = {}
    current_clause = None

    for word, lid in zip(words, labels):
        label = LABEL_MAP.get(lid, "other")

        if label == "heading" and re.match(r"^\d+(\.\d+)*", word):
            current_clause = word
            clause_graph[current_clause] = []
        elif current_clause:
            clause_graph[current_clause].append(word)

    return clause_graph


def layout_and_structure(blocks, page_image):
    encoding, words = prepare_layout_inputs(blocks, page_image)

    with torch.no_grad():
        outputs = model(**encoding)

    labels = outputs.logits.argmax(-1).squeeze().tolist()
    return build_clause_graph(words, labels)
