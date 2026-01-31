import json
from ingestion.ingestion import ingest_pdf
from orchestration.orchestrator import contract_graph
from PIL import Image

blocks = ingest_pdf("data/raw/sample_contract_form.pdf")
page_image = Image.open("data/images/page0.png")

state = {
    "blocks": blocks,
    "page_image": page_image,
    "clause_graph": {},
    "schema": {},
    "output": {}
}

final_state = contract_graph.invoke(state)

with open("data/outputs/nda_form_output.json", "w") as f:
    json.dump(final_state["output"], f, indent=2)

print(final_state["output"])
