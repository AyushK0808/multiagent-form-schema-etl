from utils.form import FormInstance
from extraction.extraction import extract_field_value

def populate_form(clause_graph, schema):
    form = FormInstance(schema)

    for field, meta in schema["fields"].items():
        section = meta["section"]
        clause_text = " ".join(clause_graph.get(section, []))
        value = extract_field_value(field, clause_text)
        form.fill(field, value)

    return form
