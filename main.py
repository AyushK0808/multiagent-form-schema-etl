"""
Simplified main entrypoint: run the full pipeline on a single NDA PDF

Behavior:
- Looks for `data/raw/NDA.pdf`; if not found, falls back to
  `data/raw/sample_contract_form.pdf` if available.
- Runs the existing pipeline and writes output to configured output dir.
"""
import json
import logging
import sys
from pathlib import Path

from config.config import get_config, update_config
from ingestion.ingestion import ingest_pdf
from orchestration.orchestrator import get_orchestrator

from PIL import Image
import fitz

# Minimal logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_directories():
    cfg = get_config()
    for p in [cfg.paths.raw_dir, cfg.paths.output_dir, cfg.paths.schema_dir, cfg.paths.test_dir]:
        p.mkdir(parents=True, exist_ok=True)


def extract_contract(pdf_path: Path, output_path: Path | None = None, form_name: str = "NDA_Form") -> dict:
    logger.info(f"Processing: {pdf_path}")

    # Ingest PDF
    logger.info("Ingesting PDF...")
    blocks, metadata = ingest_pdf(str(pdf_path))

    # Load first page image for layout analysis
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(0)
    pix = page.get_pixmap()
    page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    state = {
        "blocks": blocks,
        "page_image": page_image,
        "pdf_metadata": metadata,
        "clause_graph": {},
        "schema": {},
        "output": {}
    }

    orchestrator = get_orchestrator()
    final_state = orchestrator.process(state)

    output_data = final_state.get("output", {})

    if output_path is None:
        cfg = get_config()
        output_path = cfg.paths.output_dir / f"extracted_{pdf_path.stem}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved extraction to: {output_path}")
    return output_data


def main():
    # Ensure directories
    setup_directories()

    raw_dir = Path("data") / "raw"
    nda_path = raw_dir / "NDA.pdf"
    sample_path = raw_dir / "sample_contract_form.pdf"

    if nda_path.exists():
        pdf_to_process = nda_path
    elif sample_path.exists():
        logger.info("NDA.pdf not found, falling back to sample_contract_form.pdf")
        pdf_to_process = sample_path
    else:
        logger.error("No NDA found in data/raw. Please add data/raw/NDA.pdf")
        sys.exit(1)

    try:
        result = extract_contract(pdf_to_process)
        print(json.dumps(result.get("fields", {}), indent=2))
    except Exception as e:
        logger.exception("Extraction failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
Main entry point for contract extraction system.
"""
import json
import fitz
from PIL import Image
import argparse
import logging
from pathlib import Path
import sys

from config.config import get_config, update_config
from ingestion.ingestion import ingest_pdf
from orchestration.orchestrator import get_orchestrator
from evaluation.evaluator import Evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('contract_extraction.log')
    ]
)

logger = logging.getLogger(__name__)


def setup_directories():
    """Ensure all required directories exist."""
    config = get_config()
    for dir_path in [config.paths.raw_dir, config.paths.output_dir, 
                     config.paths.schema_dir, config.paths.test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)


def extract_contract(pdf_path: str, output_path: str = None, 
                    form_name: str = "NDA_Form") -> dict:
    """
    Extract structured data from a contract PDF.
    
    Args:
        pdf_path: Path to PDF file
        output_path: Optional output path for JSON
        form_name: Form schema to use
        
    Returns:
        Extraction result dictionary
    """
    logger.info(f"Processing: {pdf_path}")
    
    # Ingest PDF
    logger.info("Step 1/4: Ingesting PDF...")
    blocks, metadata = ingest_pdf(pdf_path)
    logger.info(f"Extracted {len(blocks)} text blocks")
    
    # Get page image for layout analysis
    logger.info("Step 2/4: Loading page image...")
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # Use first page for layout
    pix = page.get_pixmap()
    page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    
    # Build initial state
    state = {
        "blocks": blocks,
        "page_image": page_image,
        "pdf_metadata": metadata,
        "clause_graph": {},
        "schema": {},
        "output": {}
    }
    
    # Run pipeline
    logger.info("Step 3/4: Running extraction pipeline...")
    orchestrator = get_orchestrator()
    final_state = orchestrator.process(state)
    
    # Save output
    logger.info("Step 4/4: Saving output...")
    output_data = final_state["output"]
    
    if output_path:
        output_file = Path(output_path)
    else:
        config = get_config()
        output_file = config.paths.output_dir / f"extracted_{Path(pdf_path).stem}.json"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Output saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Form: {output_data.get('form', 'Unknown')}")
    print(f"Complete: {output_data.get('is_complete', False)}")
    print(f"Fields extracted: {len(output_data.get('fields', {}))}")
    
    errors = output_data.get('pipeline_metadata', {}).get('errors', [])
    if errors:
        print(f"\nErrors: {len(errors)}")
        for error in errors[:3]:
            print(f"  - {error}")
    
    warnings = output_data.get('pipeline_metadata', {}).get('warnings', [])
    if warnings:
        print(f"\nWarnings: {len(warnings)}")
        for warning in warnings[:3]:
            print(f"  - {warning}")
    
    print("="*60 + "\n")
    
    return output_data


def evaluate_system(test_dir: str):
    """
    Run evaluation on test set.
    
    Args:
        test_dir: Directory containing test cases
    """
    logger.info(f"Running evaluation on: {test_dir}")
    
    evaluator = Evaluator(test_data_dir=Path(test_dir))
    test_cases = evaluator.load_test_set()
    
    if not test_cases:
        logger.error("No test cases found")
        return
    
    logger.info(f"Found {len(test_cases)} test cases")
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"Evaluating test case {i}/{len(test_cases)}")
        
        # Define extraction function for this test case
        def extraction_fn(tc):
            pdf_path = tc.get("pdf_path")
            result = extract_contract(pdf_path)
            return result.get("fields", {})
        
        comparison = evaluator.compare_with_baseline(test_case, extraction_fn)
        results.append(comparison)
    
    # Generate report
    report = evaluator.generate_report(results)
    print("\n" + report)
    
    # Save results
    config = get_config()
    results_path = config.paths.output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Detailed results saved to: {results_path}")


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description="Contract Extraction System - Schema-guided document processing"
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input PDF file path'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file path (optional)'
    )
    
    parser.add_argument(
        '--form',
        type=str,
        default='NDA_Form',
        help='Form schema name (default: NDA_Form)'
    )
    
    parser.add_argument(
        '--evaluate',
        type=str,
        help='Run evaluation on test directory'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Disable field validation'
    )
    
    args = parser.parse_args()
    
    # Update config based on args
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        update_config(verbose=True)
    
    if args.no_validation:
        update_config(enable_validation=False)
    
    # Setup
    setup_directories()
    
    # Run evaluation or extraction
    if args.evaluate:
        evaluate_system(args.evaluate)
    elif args.input:
        try:
            result = extract_contract(args.input, args.output, args.form)
            print("\nExtracted Fields:")
            print(json.dumps(result.get("fields", {}), indent=2))
        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()