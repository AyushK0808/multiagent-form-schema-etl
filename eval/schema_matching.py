"""
eval/schema_matching.py
=======================
Schema-matching evaluation. Eight schemas are registered; 50 labelled queries
(field-name sets as an extractor would return them) are matched against the
registry. 40 queries belong to a registered schema (exact names, synonym
renames, subsets, subsets with unrelated extra fields) and 10 describe document
types that are not registered and must be rejected.

Prediction = top-1 schema if its cosine similarity >= sigma, else "no match".
Precision = correct matches / predicted matches; recall = correct matches /
queries that have a registered schema. Sweeping sigma gives the threshold curve.

Usage: python -m eval.schema_matching [--out results_final/schema_matching.json]
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from eval.data import CORD_SCHEMA, SROIE_SCHEMA
from eval.make_synthetic import SCHEMAS as SYNTH_SCHEMAS
from schema.schema_registry import SchemaRegistry

SIGMAS = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]
OPERATING_SIGMA = 0.70

PURCHASE_ORDER = {"form_name": "purchase_order", "fields": {
    "po_number": {"type": "string", "description": "Purchase order identifier"},
    "order_date": {"type": "date", "description": "Date the order was placed"},
    "supplier": {"type": "string", "description": "Company supplying the goods"},
    "ship_to": {"type": "string", "description": "Delivery address"},
    "order_total": {"type": "number", "description": "Total value of the order"},
}}
FUNSD_FORM = {"form_name": "funsd_form", "fields": {
    "question": {"type": "string", "description": "Printed label of a form entry"},
    "answer": {"type": "string", "description": "Filled-in value of a form entry"},
    "header": {"type": "string", "description": "Section header of the form"},
}}
REGISTERED = [SYNTH_SCHEMAS["nda"], SYNTH_SCHEMAS["employment"], SYNTH_SCHEMAS["invoice"],
              SYNTH_SCHEMAS["deed"], SROIE_SCHEMA, CORD_SCHEMA, PURCHASE_ORDER, FUNSD_FORM]

# (label, field names). Variants per schema: exact, synonyms, subset, subset + noise, synonyms + noise.
POSITIVES: Dict[str, List[List[str]]] = {
    "nda": [
        ["effective_date", "disclosing_party", "receiving_party", "governing_law", "confidentiality_years"],
        ["agreement_date", "discloser", "recipient", "jurisdiction", "confidentiality_period"],
        ["effective_date", "disclosing_party", "receiving_party"],
        ["disclosing_party", "receiving_party", "governing_law", "page_count"],
        ["start_date", "disclosing_company", "receiving_company", "applicable_law"],
    ],
    "employment_contract": [
        ["employee_name", "employer", "start_date", "job_title", "salary"],
        ["worker_name", "company", "commencement_date", "position", "annual_pay"],
        ["employee_name", "job_title", "salary"],
        ["employee_name", "employer", "salary", "document_language"],
        ["staff_member", "hiring_company", "date_of_joining", "role", "gross_salary"],
    ],
    "invoice": [
        ["invoice_number", "invoice_date", "vendor", "customer", "total_amount"],
        ["invoice_no", "issue_date", "supplier_name", "client_name", "amount_due"],
        ["invoice_number", "vendor", "total_amount"],
        ["invoice_date", "customer", "total_amount", "currency"],
        ["bill_number", "billing_date", "seller", "buyer", "grand_total"],
    ],
    "land_purchase_agreement": [
        ["seller", "buyer", "property_address", "purchase_price", "closing_date"],
        ["vendor_name", "purchaser", "land_location", "sale_price", "completion_date"],
        ["seller", "buyer", "purchase_price"],
        ["property_address", "purchase_price", "closing_date", "witness_name"],
        ["grantor", "grantee", "parcel_address", "consideration_amount", "settlement_date"],
    ],
    "sroie_receipt": [
        ["company", "date", "address", "total"],
        ["store_name", "receipt_date", "store_address", "amount_paid"],
        ["company", "total"],
        ["company", "date", "address", "total", "cashier"],
        ["merchant", "purchase_date", "merchant_address", "total_paid"],
    ],
    "cord_receipt": [
        ["subtotal_price", "tax_price", "total_price", "cashprice", "changeprice", "creditcardprice"],
        ["subtotal", "tax", "total", "cash_paid", "change_returned", "card_payment"],
        ["subtotal_price", "tax_price", "total_price"],
        ["total_price", "cashprice", "changeprice", "table_number"],
        ["sub_total", "vat", "grand_total", "cash_tendered", "change_due"],
    ],
    "purchase_order": [
        ["po_number", "order_date", "supplier", "ship_to", "order_total"],
        ["order_number", "date_ordered", "vendor_name", "delivery_address", "total_value"],
        ["po_number", "supplier", "order_total"],
        ["order_date", "ship_to", "order_total", "buyer_contact"],
        ["purchase_order_id", "placed_on", "supplier_company", "shipping_address", "order_amount"],
    ],
    "funsd_form": [
        ["question", "answer", "header"],
        ["label", "value", "section_title"],
        ["question", "answer"],
        ["question", "answer", "header", "page_number"],
        ["field_label", "field_value", "heading"],
    ],
}
NEGATIVES: List[List[str]] = [
    ["patient_name", "diagnosis", "prescribed_medication", "dosage", "physician"],
    ["flight_number", "departure_airport", "arrival_airport", "boarding_time", "seat"],
    ["student_name", "course_title", "grade", "credit_hours", "semester"],
    ["vehicle_vin", "make", "model", "odometer_reading", "registration_state"],
    ["account_holder", "iban", "statement_period", "opening_balance", "closing_balance"],
    ["hotel_name", "check_in_date", "check_out_date", "room_type", "guest_name"],
    ["policy_number", "insured_name", "coverage_amount", "premium", "expiry_date"],
    ["recipe_title", "ingredients", "cooking_time", "servings", "oven_temperature"],
    ["tournament_name", "team_a", "team_b", "final_score", "venue"],
    ["species", "habitat", "wingspan", "conservation_status", "region"],
]


def build_queries() -> List[Tuple[List[str], Optional[str]]]:
    queries = [(fields, label) for label, variants in POSITIVES.items() for fields in variants]
    queries += [(fields, None) for fields in NEGATIVES]
    assert len(queries) == 50, len(queries)
    return queries


def evaluate(sigmas: List[float]) -> Dict:
    with tempfile.TemporaryDirectory() as tmp:
        registry = SchemaRegistry(registry_dir=Path(tmp), sim_threshold=0.0)
        for schema in REGISTERED:
            registry.register(schema)
        top: List[Tuple[Optional[str], float]] = []
        queries = build_queries()
        for fields, _ in queries:
            hits = registry.find_similar({f: None for f in fields}, top_k=1)
            top.append((hits[0][2]["form_name"], hits[0][0]) if hits else (None, 0.0))

    n_pos = sum(1 for _, label in queries if label is not None)
    curve = []
    for sigma in sigmas:
        predicted = correct = false_accept = 0
        for (pred, sim), (_, label) in zip(top, queries):
            if pred is None or sim < sigma:
                continue
            predicted += 1
            correct += int(pred == label)
            false_accept += int(label is None)
        curve.append({
            "sigma": sigma,
            "precision": correct / predicted if predicted else 1.0,
            "recall": correct / n_pos,
            "false_accept_rate": false_accept / (len(queries) - n_pos),
        })
    return {"n_queries": len(queries), "n_positive": n_pos, "n_negative": len(queries) - n_pos,
            "n_registered": len(REGISTERED), "operating_sigma": OPERATING_SIGMA, "curve": curve,
            "top1": [{"fields": f, "label": l, "pred": p, "sim": round(s, 3)}
                     for (f, l), (p, s) in zip(queries, top)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results_final") / "schema_matching.json")
    args = ap.parse_args()
    result = evaluate(SIGMAS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{'sigma':>6} {'P':>6} {'R':>6} {'FAR':>6}")
    for row in result["curve"]:
        mark = "  <- operating point" if row["sigma"] == OPERATING_SIGMA else ""
        print(f"{row['sigma']:>6.2f} {row['precision']:>6.3f} {row['recall']:>6.3f} "
              f"{row['false_accept_rate']:>6.3f}{mark}")


if __name__ == "__main__":
    main()
