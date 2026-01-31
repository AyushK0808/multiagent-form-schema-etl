from langgraph.graph import StateGraph
from typing import TypedDict
from schema.schema import load_schema
from extraction.form_filler import populate_form
from layout_analysis.layout_structure import layout_and_structure

class ContractState(TypedDict):
    blocks: list
    page_image: object
    clause_graph: dict
    schema: dict
    output: dict


def layout_node(state):
    state["clause_graph"] = layout_and_structure(
        state["blocks"], state["page_image"]
    )
    return state


def schema_node(state):
    state["schema"] = load_schema()
    return state


def extraction_node(state):
    form = populate_form(state["clause_graph"], state["schema"])
    state["output"] = form.to_json()
    return state


graph = StateGraph(ContractState)
graph.add_node("layout", layout_node)
graph.add_node("schema", schema_node)
graph.add_node("extract", extraction_node)

graph.set_entry_point("layout")
graph.add_edge("layout", "schema")
graph.add_edge("schema", "extract")

contract_graph = graph.compile()
