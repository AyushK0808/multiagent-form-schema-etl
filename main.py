import json
import fitz  # PyMuPDF
from PIL import Image
import io
import os

from ingestion.ingestion import ingest_pdf
from orchestration.orchestrator import contract_graph

# Ensure output directories exist
os.makedirs("data/outputs", exist_ok=True)

PDF_PATH = "data/raw/sample_contract_form.pdf"

print(f"Ingesting {PDF_PATH}...")
blocks = ingest_pdf(PDF_PATH)

# Fix: Extract the first page as an image directly from PDF bytes
# instead of expecting a file on disk.
doc = fitz.open(PDF_PATH)
page = doc.load_page(0)
pix = page.get_pixmap()
# Convert PyMuPDF Pixmap to PIL Image
page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

state = {
    "blocks": blocks,
    "page_image": page_image,
    "clause_graph": {},
    "schema": {},
    "output": {}
}

print("Running pipeline...")
try:
    final_state = contract_graph.invoke(state)
    
    output_path = "data/outputs/nda_form_output.json"
    with open(output_path, "w") as f:
        json.dump(final_state["output"], f, indent=2)

    print(f"Success! Output saved to {output_path}")
    print(json.dumps(final_state["output"], indent=2))
    
except Exception as e:
    print(f"Pipeline failed: {e}")
    import traceback
    traceback.print_exc()