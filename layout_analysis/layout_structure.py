# layout_structure.py
import torch
import re
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from typing import List, Dict

processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False
)

model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base"
)
model.eval()
