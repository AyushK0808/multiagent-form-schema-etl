"""
eval/make_synthetic.py
======================
Regenerates the multi-page synthetic stress set. Faker fills NDA, employment
contract, invoice and land-deed templates, ReportLab renders them, and each page
is rasterised and degraded (rotation, blur, noise, JPEG) to look scanned. The
generator writes the gold JSON, so the ground truth is exact by construction.

Fields sit on different pages and amounts appear both in the body and in a
summary block, so page-by-page extraction and merging are exercised.

Usage: python -m eval.make_synthetic [--n-docs 15] [--seed 7] [--out data/eval/synth]
"""
from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import fitz
from faker import Faker
from PIL import Image, ImageFilter
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table

DEFAULT_OUT = Path("data") / "eval" / "synth"


def _f(ftype: str, description: str) -> Dict[str, Any]:
    return {"type": ftype, "description": description, "required": True}


SCHEMAS = {
    "nda": {"form_name": "nda", "fields": {
        "effective_date": _f("date", "Date the agreement takes effect"),
        "disclosing_party": _f("string", "Party disclosing confidential information"),
        "receiving_party": _f("string", "Party receiving confidential information"),
        "governing_law": _f("string", "Jurisdiction whose law governs the agreement"),
        "confidentiality_years": _f("number", "Number of years the confidentiality obligation lasts"),
    }},
    "employment": {"form_name": "employment_contract", "fields": {
        "employee_name": _f("string", "Full name of the employee"),
        "employer": _f("string", "Name of the employing company"),
        "start_date": _f("date", "First day of employment"),
        "job_title": _f("string", "Position the employee is hired for"),
        "salary": _f("number", "Annual gross salary in dollars"),
    }},
    "invoice": {"form_name": "invoice", "fields": {
        "invoice_number": _f("string", "Invoice identifier"),
        "invoice_date": _f("date", "Date the invoice was issued"),
        "vendor": _f("string", "Company issuing the invoice"),
        "customer": _f("string", "Company being billed"),
        "total_amount": _f("number", "Total amount due in dollars"),
    }},
    "deed": {"form_name": "land_purchase_agreement", "fields": {
        "seller": _f("string", "Name of the seller"),
        "buyer": _f("string", "Name of the buyer"),
        "property_address": _f("string", "Address of the property being sold"),
        "purchase_price": _f("number", "Agreed purchase price in dollars"),
        "closing_date": _f("date", "Date the sale closes"),
    }},
}

_FILLER = [
    "The parties acknowledge that the terms set out herein are the entire understanding between them.",
    "Notices under this document must be given in writing and delivered to the addresses recorded below.",
    "No amendment is effective unless made in writing and signed by an authorised representative of each party.",
    "If any provision is held unenforceable, the remaining provisions continue in full force.",
    "Headings are for convenience only and do not affect the interpretation of this document.",
]


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _date(d) -> str:
    return d.strftime("%B %d, %Y")


def _paras(rng: random.Random, n: int) -> List[str]:
    return [rng.choice(_FILLER) for _ in range(n)]


def _nda(fk: Faker, rng: random.Random) -> Tuple[List[List[str]], Dict[str, Any]]:
    eff, disc, recv = fk.date_between("-2y", "today"), fk.company(), fk.company()
    law, years = rng.choice(["Delaware", "New York", "California", "Texas"]), rng.choice([2, 3, 5, 7])
    p1 = [f"MUTUAL NON-DISCLOSURE AGREEMENT",
          f"This Agreement is made effective as of {_date(eff)} between {disc} (the Disclosing Party) "
          f"and {recv} (the Receiving Party)."] + _paras(rng, 3)
    p2 = ["2. Obligations."] + _paras(rng, 4) + [
        f"The Receiving Party shall keep all Confidential Information secret for a period of {years} years."]
    p3 = ["3. General."] + _paras(rng, 2) + [f"This Agreement is governed by the laws of the State of {law}."]
    gold = {"effective_date": eff.isoformat(), "disclosing_party": disc, "receiving_party": recv,
            "governing_law": law, "confidentiality_years": str(years)}
    return [p1, p2, p3], gold


def _employment(fk: Faker, rng: random.Random):
    name, emp, start = fk.name(), fk.company(), fk.date_between("-1y", "+6m")
    title, salary = fk.job(), rng.randrange(45, 190) * 1000
    p1 = ["EMPLOYMENT CONTRACT", f"This contract is between {emp} (the Employer) and {name} (the Employee)."] \
        + _paras(rng, 3)
    p2 = [f"1. Position. The Employee is engaged as {title}.",
          f"2. Commencement. Employment begins on {_date(start)}."] + _paras(rng, 3)
    p3 = [f"3. Compensation. The Employer shall pay an annual gross salary of {_money(salary)}."] \
        + _paras(rng, 2)
    gold = {"employee_name": name, "employer": emp, "start_date": start.isoformat(),
            "job_title": title, "salary": str(salary)}
    return [p1, p2, p3], gold


def _invoice(fk: Faker, rng: random.Random):
    num, d = f"INV-{rng.randrange(10000, 99999)}", fk.date_between("-1y", "today")
    vendor, cust = fk.company(), fk.company()
    items = [(fk.bs().title(), rng.randrange(1, 9), rng.randrange(20, 900)) for _ in range(rng.randrange(4, 9))]
    total = float(sum(q * p for _, q, p in items))
    p1 = [f"INVOICE {num}", f"Issued {_date(d)} by {vendor}", f"Billed to: {cust}", fk.address().replace("\n", ", ")]
    lines = [f"{desc} - {q} x {_money(p)} = {_money(q * p)}" for desc, q, p in items]
    p2 = ["Line items"] + lines
    p3 = ["Summary", f"Total amount due: {_money(total)}", "Payment is due within 30 days."]
    gold = {"invoice_number": num, "invoice_date": d.isoformat(), "vendor": vendor,
            "customer": cust, "total_amount": f"{total:.2f}"}
    return [p1, p2, p3], gold


def _deed(fk: Faker, rng: random.Random):
    seller, buyer, addr = fk.name(), fk.name(), fk.address().replace("\n", ", ")
    price, close = rng.randrange(80, 900) * 1000, fk.date_between("today", "+1y")
    p1 = ["LAND PURCHASE AGREEMENT", f"Seller: {seller}", f"Buyer: {buyer}",
          f"Property: {addr}"] + _paras(rng, 2)
    p2 = [f"The Seller agrees to sell and the Buyer agrees to buy the Property for {_money(price)}."] \
        + _paras(rng, 4)
    p3 = [f"Closing shall take place on {_date(close)}."] + _paras(rng, 2)
    gold = {"seller": seller, "buyer": buyer, "property_address": addr,
            "purchase_price": str(price), "closing_date": close.isoformat()}
    return [p1, p2, p3], gold


BUILDERS: Dict[str, Callable] = {"nda": _nda, "employment": _employment, "invoice": _invoice, "deed": _deed}


def _render_pdf(pages: List[List[str]], path: Path) -> None:
    styles = getSampleStyleSheet()
    story: List[Any] = []
    for i, page in enumerate(pages):
        for j, text in enumerate(page):
            story.append(Paragraph(text, styles["Title"] if i == 0 and j == 0 else styles["BodyText"]))
            story.append(Spacer(1, 10))
        if i < len(pages) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def _scan(pdf_path: Path, rng: random.Random) -> None:
    """Rasterise each page and apply scanner-like degradation, then rebuild the PDF."""
    src = fitz.open(str(pdf_path))
    out = fitz.open()
    for page in src:
        img = Image.open(io.BytesIO(page.get_pixmap(dpi=150).tobytes("png"))).convert("L")
        img = img.rotate(rng.uniform(-1.5, 1.5), expand=False, fillcolor=255)
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
        noise = Image.effect_noise(img.size, rng.uniform(8, 18))
        img = Image.blend(img, noise, 0.08)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=rng.randrange(55, 80))
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=buf.getvalue())
    src.close()
    out.save(str(pdf_path))
    out.close()


def generate(out_dir: Path, n_docs: int, seed: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    fk = Faker("en_US")
    Faker.seed(seed)
    rng = random.Random(seed)
    count = 0
    kinds = list(BUILDERS)
    for n in range(n_docs):  # round-robin over document types
        kind = kinds[n % len(kinds)]
        pages, gold = BUILDERS[kind](fk, rng)
        stem = f"{kind}_{n // len(kinds):02d}"
        pdf = out_dir / f"{stem}.pdf"
        _render_pdf(pages, pdf)
        _scan(pdf, rng)
        (out_dir / f"{stem}.json").write_text(json.dumps(
            {"pdf": pdf.name, "gold": gold, "schema": SCHEMAS[kind], "n_pages": len(pages)},
            indent=2), encoding="utf-8")
        count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    print(f"generated {generate(args.out, args.n_docs, args.seed)} documents in {args.out}")


if __name__ == "__main__":
    main()
