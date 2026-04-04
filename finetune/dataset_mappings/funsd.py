"""
FUNSD-specific label mapping helpers.
"""
from __future__ import annotations


FUNSD_BIO_LABELS = [
    "O",
    "B-QUESTION",
    "I-QUESTION",
    "B-ANSWER",
    "I-ANSWER",
    "B-HEADER",
    "I-HEADER",
]

FUNSD_NER_ID_TO_LABEL = {
    0: "O",
    1: "B-HEADER",
    2: "I-HEADER",
    3: "B-QUESTION",
    4: "I-QUESTION",
    5: "B-ANSWER",
    6: "I-ANSWER",
}


def funsd_ner_tag_to_label(tag: int | str) -> str:
    if isinstance(tag, str):
        text = tag.strip().upper()
        return text if text in FUNSD_BIO_LABELS else "O"
    if isinstance(tag, int):
        return FUNSD_NER_ID_TO_LABEL.get(tag, "O")
    return "O"


def funsd_word_bio_label(entity_label: str, word_index: int) -> str:
    base = str(entity_label or "other").strip().lower()
    if base == "other":
        return "O"

    if base == "question":
        prefix = "QUESTION"
    elif base == "answer":
        prefix = "ANSWER"
    elif base == "header":
        prefix = "HEADER"
    else:
        return "O"

    return f"B-{prefix}" if word_index == 0 else f"I-{prefix}"
