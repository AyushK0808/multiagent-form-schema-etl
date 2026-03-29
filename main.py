"""
Agentic ETL Fabric -- main entry point.

Usage
-----
    python main.py --pdf data/raw/sample_contract_form.pdf --form invoice_schema
    python main.py --pdf contract.pdf --schema-id <schema_uuid>
    python main.py --pdf contract.pdf --no-schema-agent   # offline mode
    python main.py --pdf contract.pdf --no-donut          # LayoutLM only
    python main.py --list-schemas
    python main.py --query-db invoice_schema

Environment variables
---------------------
    GROQ_API_KEY   -- required for schema-resolution agent
"""
import argparse
import json
import sys
from pathlib import Path

import fitz
from PIL import Image

from config.config import get_config, update_config

# Configure logging before any other imports that may call getLogger().
# utils/logging_setup.py forces UTF-8 on the stdout handler so that
# non-ASCII log messages (from other modules) do not crash on Windows.

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory bootstrap
# ---------------------------------------------------------------------------

def setup_directories() -> None:
    cfg = get_config()
    for p in [cfg.paths.raw_dir, cfg.paths.output_dir, cfg.paths.schema_dir,
              cfg.paths.test_dir, cfg.paths.registry_dir]:
        p.mkdir(parents=True, exist_ok=True)
    (Path("data") / "intermediate").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Core pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    pdf_path: Path,
    form_name: "str | None" = None,
    schema_id: "str | None" = None,
    output_path: "Path | None" = None,
) -> dict:
    """Run the full agentic ETL pipeline. Returns the final output dict."""
    from ingestion.ingestion import ingest_pdf
    from orchestration.orchestrator import get_orchestrator
    from schema.schema import load_schema

    logger.info("Processing PDF: %s", pdf_path)

    # 1. Ingest
    blocks, metadata = ingest_pdf(str(pdf_path))
    logger.info("Ingested %d blocks from %d pages", len(blocks), metadata["total_pages"])

    # 2. First-page image
    doc  = fitz.open(str(pdf_path))
    page = doc.load_page(0)
    pix  = page.get_pixmap()
    page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    # 3. Load the nominal schema (schema_resolve node may replace it)
    if not form_name and not schema_id:
        cfg = get_config()
        if not cfg.enable_schema_recognition:
            raise ValueError("A schema must be provided via --form or --schema-id")
        from schema.schema_recognizer import SchemaRecognizer

        recognizer = SchemaRecognizer(
            layout_model_path=cfg.model.schema_recognition_layout_model,
            donut_model_path=cfg.model.schema_recognition_donut_model,
        )
        prediction = recognizer.predict(page_image)
        form_name = prediction["schema_name"]
        logger.info(
            "Auto-recognized schema: %s via %s (confidence=%.3f)",
            form_name,
            prediction["source"],
            prediction["confidence"],
        )

    schema = load_schema(form_name=form_name, schema_id=schema_id)

    # 4. Build initial pipeline state
    state = {
        "blocks":       blocks,
        "page_image":   page_image,
        "pdf_metadata": {**metadata, "source_path": str(pdf_path)},
        "schema_recognition": {"form_name": form_name, "schema_id": schema_id},
        "schema":       schema,
        "clause_graph": {},
        "output":       {},
        "errors":       [],
        "warnings":     [],
    }

    # 5. Run pipeline
    orchestrator = get_orchestrator()
    final_state  = orchestrator.process(state)
    output_data  = final_state.get("output", {})

    # 6. Save output (explicit UTF-8 so non-ASCII values in field data are safe)
    if output_path is None:
        cfg = get_config()
        output_path = cfg.paths.output_dir / f"extracted_{pdf_path.stem}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Output written to: %s", output_path)

    # 7. Save intermediate phase summaries
    _save_intermediates(final_state)

    # 8. Print summary (ASCII-only to avoid console encoding errors)
    _print_summary(output_data)
    return output_data


# ---------------------------------------------------------------------------
# Utility commands
# ---------------------------------------------------------------------------

def list_schemas() -> None:
    from schema.schema import SchemaManager

    manager = SchemaManager()
    schemas = manager.list_schemas()
    if not schemas:
        print("Schema store is empty. Insert schemas into SQLite before running the pipeline.")
        return
    print(f"\n{'Schema ID':<38}  {'Form Name':<30}  {'Version':<8}  Updated")
    print("-" * 96)
    for entry in schemas:
        print(
            f"{entry['schema_id']:<38}  "
            f"{entry['form_name']:<30}  "
            f"{entry.get('version', ''):<8}  "
            f"{entry.get('updated_at', '')[:19]}"
        )


def query_db(form_name: str, limit: int = 20) -> None:
    from database.db_manager import DatabaseManager
    db   = DatabaseManager()
    rows = db.query_records(form_name, limit=limit)
    if not rows:
        print(f"No records for '{form_name}'")
        return
    print(f"\n{len(rows)} record(s) in '{form_name}':\n")
    for row in rows:
        print(json.dumps(dict(row), indent=2, default=str))
        print("-" * 60)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_intermediates(state: dict) -> None:
    inter  = Path("data") / "intermediate"
    bundle = state.get("bundle")

    def _cov(result):
        return getattr(result, "field_coverage", None) if result else None

    phases = {
        "01_ingestion.json":        {"num_blocks": len(state.get("blocks", []))},
        "02_layout.json":           {"num_clauses": len(state.get("clause_graph", {}))},
        "03_parallel_extract.json": {
            "donut_coverage":    _cov(bundle.donut)    if bundle else None,
            "layoutlm_coverage": _cov(bundle.layoutlm) if bundle else None,
        },
        "04_policy_fuse.json": {
            "coverage":       getattr(state.get("policy_result"), "coverage",       None),
            "consistency_ok": getattr(state.get("policy_result"), "consistency_ok", None),
            "issues":         getattr(state.get("policy_result"), "issues",         []),
        },
        "05_schema_resolve.json": {
            "resolved_schema": state.get("resolved_schema", {}).get("form_name"),
            "schema_id":       state.get("schema_id"),
            "field_mapping":   state.get("field_mapping"),
        },
        "06_db.json": {"record_id": state.get("record_id")},
    }
    for fname, data in phases.items():
        (inter / fname).write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )


def _print_summary(output: dict) -> None:
    """Print extraction summary using only ASCII printable characters."""
    fields = output.get("fields", {})
    pm     = output.get("pipeline_metadata", {})
    errors = pm.get("errors", [])
    warns  = pm.get("warnings", [])

    sep = "=" * 60
    print(f"\n{sep}")
    print("  AGENTIC ETL -- EXTRACTION SUMMARY")
    print(sep)

    def _trunc(s, n=16):
        s = str(s)
        return s[:n] + "..." if len(s) > n else s

    print(f"  Form        : {output.get('form', '?')}")
    print(f"  Schema ID   : {_trunc(output.get('schema_id', '?'))}")
    print(f"  Record ID   : {_trunc(output.get('record_id', '?'))}")
    print(f"  Complete    : {output.get('is_complete', False)}")
    print(f"  Coverage    : {pm.get('field_coverage', 0.0):.2f}")
    print(f"  Consistent  : {pm.get('consistency_ok', True)}")
    print("  Fields      :")
    for k, v in fields.items():
        conf = pm.get("confidences", {}).get(k, 0.0)
        # Encode value to ASCII, replacing anything non-printable
        v_str = str(v).encode("ascii", errors="replace").decode("ascii")
        print(f"    {k:<28} = {v_str:<30}  [conf={conf:.2f}]")
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors[:5]:
            e_str = str(e).encode("ascii", errors="replace").decode("ascii")
            print(f"    [ERROR] {e_str}")
    if warns:
        print(f"\n  Warnings ({len(warns)}):")
        for w in warns[:5]:
            w_str = str(w).encode("ascii", errors="replace").decode("ascii")
            print(f"    [WARN]  {w_str}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic ETL Fabric for semi-structured document extraction"
    )
    parser.add_argument("--pdf",             type=Path, help="Path to input PDF")
    parser.add_argument("--form",
                        help="Schema form name stored in SQLite")
    parser.add_argument("--schema-id",
                        help="Schema ID stored in SQLite")
    parser.add_argument("--output",          type=Path, help="JSON output path")
    parser.add_argument("--no-schema-agent", action="store_true",
                        help="Disable Groq schema-resolution agent (offline mode)")
    parser.add_argument("--no-donut",        action="store_true",
                        help="Disable Donut extractor (LayoutLM only, faster)")
    parser.add_argument("--no-db",           action="store_true",
                        help="Disable database population")
    parser.add_argument("--list-schemas",    action="store_true",
                        help="List all registered schemas and exit")
    parser.add_argument("--query-db",        metavar="FORM_NAME",
                        help="Print DB records for a form name and exit")
    parser.add_argument("--verbose",         action="store_true",
                        help="Enable DEBUG logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    setup_directories()

    # -- Utility commands ----------------------------------------------------
    if args.list_schemas:
        list_schemas()
        return
    if args.query_db:
        query_db(args.query_db)
        return

    # -- Feature flags -------------------------------------------------------
    if args.no_schema_agent:
        update_config(enable_schema_agent=False)
    if args.no_donut:
        update_config(enable_parallel_extraction=False)
    if args.no_db:
        update_config(enable_db_population=False)

    # -- PDF resolution ------------------------------------------------------
    pdf_path = args.pdf
    if pdf_path is None:
        raw = Path("data") / "raw"
        for candidate in ("NDA.pdf", "sample_contract_form.pdf"):
            p = raw / candidate
            if p.exists():
                pdf_path = p
                break
    if pdf_path is None or not pdf_path.exists():
        logger.error("No PDF found. Use --pdf <path> or place a file in data/raw/")
        sys.exit(1)

    # -- Run -----------------------------------------------------------------
    try:
        result = run_pipeline(
            pdf_path,
            form_name=args.form,
            schema_id=args.schema_id,
            output_path=args.output,
        )
        # ensure_ascii=True keeps the final JSON print safe on any console
        print(json.dumps(result.get("fields", {}), indent=2, ensure_ascii=True))
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
