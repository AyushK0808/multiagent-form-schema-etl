"""
Dataset-specific mapping helpers.
"""

from .cord import cord_layout_label, cord_parse_ground_truth, cord_word_bbox
from .docbank import docbank_token_label, DOCBANK_LABEL_MAP
from .doclaynet import doclaynet_category_name, DOCLAYNET_ID_TO_NAME
from .docvqa import docvqa_bbox, docvqa_ocr_token_label, docvqa_question_type
from .funsd import funsd_ner_tag_to_label, funsd_word_bio_label, FUNSD_BIO_LABELS, FUNSD_NER_ID_TO_LABEL
from .infographicvqa import infographicvqa_bbox, infographicvqa_ocr_token_label, infographicvqa_question_type
from .kleister_nda import kleister_nda_bbox, kleister_nda_entity_label, kleister_nda_is_spatial, KLEISTER_NDA_ENTITY_MAP, SPATIAL_FIELDS
from .rvl_cdip import rvl_cdip_label_name, RVL_CDIP_CLASSES
from .sroie import sroie_bbox_from_points, sroie_entity_label, sroie_text_label, sroie_word_label, SROIE_KEY_FIELDS, SROIE_ENTITY_LABEL
from .synthdog_en import synthdog_bbox, synthdog_parse_ground_truth, synthdog_token_label, SYNTHDOG_LABEL_MAP

__all__ = [
    # cord
    "cord_layout_label", "cord_parse_ground_truth", "cord_word_bbox",
    # docbank
    "docbank_token_label", "DOCBANK_LABEL_MAP",
    # doclaynet
    "doclaynet_category_name", "DOCLAYNET_ID_TO_NAME",
    # docvqa
    "docvqa_bbox", "docvqa_ocr_token_label", "docvqa_question_type",
    # funsd
    "funsd_ner_tag_to_label", "funsd_word_bio_label", "FUNSD_BIO_LABELS", "FUNSD_NER_ID_TO_LABEL",
    # infographicvqa
    "infographicvqa_bbox", "infographicvqa_ocr_token_label", "infographicvqa_question_type",
    # kleister_nda
    "kleister_nda_bbox", "kleister_nda_entity_label", "kleister_nda_is_spatial",
    "KLEISTER_NDA_ENTITY_MAP", "SPATIAL_FIELDS",
    # rvl_cdip
    "rvl_cdip_label_name", "RVL_CDIP_CLASSES",
    # sroie
    "sroie_bbox_from_points", "sroie_entity_label", "sroie_text_label", "sroie_word_label",
    "SROIE_KEY_FIELDS", "SROIE_ENTITY_LABEL",
    # synthdog_en
    "synthdog_bbox", "synthdog_parse_ground_truth", "synthdog_token_label", "SYNTHDOG_LABEL_MAP",
]
