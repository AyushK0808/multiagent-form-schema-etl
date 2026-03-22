"""
Orchestration pipeline using LangGraph with validation and logging.

Key change: the extraction node now passes page_image to FormFiller so the
vision-LLM fallback can fire for any fields the text LLM left null.
"""
from langgraph.graph import StateGraph
from typing import TypedDict, Optional
import logging
from datetime import datetime

from schema.schema import load_schema
from extraction.form_filler import FormFiller
from layout_analysis.layout_structure import LayoutAnalyzer
from utils.validation import ValidationRecoveryManager
from config.config import get_config

logger = logging.getLogger(__name__)


class ContractState(TypedDict):
    """State dictionary for contract processing pipeline."""
    # Input
    blocks: list
    page_image: object
    pdf_metadata: dict

    # Intermediate
    clause_graph: dict
    schema: dict
    layout_predictions: list

    # Output
    form: object
    output: dict

    # Metadata
    pipeline_start: str
    pipeline_end: str
    errors: list
    warnings: list


class ContractOrchestrator:
    """Orchestrates the contract extraction pipeline."""

    def __init__(self):
        self.config = get_config()
        self.layout_analyzer = LayoutAnalyzer()
        self.form_filler = FormFiller()
        self.validator = ValidationRecoveryManager()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(ContractState)

        graph.add_node("layout",   self._layout_node)
        graph.add_node("schema",   self._schema_node)
        graph.add_node("extract",  self._extraction_node)
        graph.add_node("validate", self._validation_node)
        graph.add_node("finalize", self._finalize_node)

        graph.set_entry_point("layout")
        graph.add_edge("layout",   "schema")
        graph.add_edge("schema",   "extract")
        graph.add_edge("extract",  "validate")
        graph.add_edge("validate", "finalize")

        return graph.compile()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _layout_node(self, state: ContractState) -> ContractState:
        logger.info("Running layout analysis...")
        try:
            result = self.layout_analyzer.analyze(
                state["blocks"], state["page_image"]
            )
            state["clause_graph"] = result["clause_graph"]
            state["layout_predictions"] = result.get("predictions", [])
            logger.info(f"Found {len(state['clause_graph'])} clauses")
        except Exception as e:
            logger.error(f"Layout analysis failed: {e}")
            state.setdefault("errors", []).append(f"Layout: {str(e)}")
            state["clause_graph"] = {}
        return state

    def _schema_node(self, state: ContractState) -> ContractState:
        logger.info("Loading schema...")
        try:
            state["schema"] = load_schema("NDA_Form")
            logger.info(f"Loaded schema: {state['schema']['form_name']}")
        except Exception as e:
            logger.error(f"Schema loading failed: {e}")
            state.setdefault("errors", []).append(f"Schema: {str(e)}")
        return state

    def _extraction_node(self, state: ContractState) -> ContractState:
        """
        Extract fields using text LLM + vision LLM fallback.

        page_image is forwarded to FormFiller so that any field the text LLM
        leaves null gets a second attempt via the local vision model.
        """
        logger.info("Extracting fields (text LLM + vision fallback)...")
        try:
            full_text = " ".join(
                str(b.get("text", "")) for b in state.get("blocks", [])
            )

            form = self.form_filler.populate(
                state["clause_graph"],
                state["schema"],
                full_text=full_text,
                page_image=state.get("page_image"),   # ← enables vision fallback
            )

            state["form"] = form
            logger.info(f"Extracted {len(form.fields)} fields")

            # Warn about required fields still set to NaN
            nan_required = [
                name
                for name, meta in state["schema"]["fields"].items()
                if meta.get("required") and form.fields.get(name) == "NaN"
            ]
            if nan_required:
                state.setdefault("warnings", []).append(
                    f"Required fields not found after all passes: "
                    f"{', '.join(nan_required)}"
                )

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            state.setdefault("errors", []).append(f"Extraction: {str(e)}")
        return state

    def _validation_node(self, state: ContractState) -> ContractState:
        logger.info("Validating fields...")
        if not self.config.enable_validation:
            logger.info("Validation disabled")
            return state
        try:
            form = state.get("form")
            if not form:
                return state

            recovered_data, remaining_errors = self.validator.validate_and_recover(
                form.fields,
                state["schema"],
                state["clause_graph"],
                page_image=state.get("page_image"),
            )

            for field, value in recovered_data.items():
                form.fill(field, value)

            if remaining_errors:
                state.setdefault("errors", []).extend(remaining_errors)
                logger.warning(f"Validation errors: {len(remaining_errors)}")
            else:
                logger.info("All validations passed")

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            state.setdefault("errors", []).append(f"Validation: {str(e)}")
        return state

    def _finalize_node(self, state: ContractState) -> ContractState:
        logger.info("Finalizing output...")
        form = state.get("form")
        if form:
            output = form.to_dict()
            output["pipeline_metadata"] = {
                "start_time":   state.get("pipeline_start"),
                "end_time":     datetime.now().isoformat(),
                "num_clauses":  len(state.get("clause_graph", {})),
                "errors":       state.get("errors", []),
                "warnings":     state.get("warnings", []),
                "pdf_metadata": state.get("pdf_metadata", {}),
            }
            state["output"] = output
        else:
            state["output"] = {"error": "No form generated"}

        state["pipeline_end"] = datetime.now().isoformat()
        return state

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def process(self, state: ContractState) -> ContractState:
        state["pipeline_start"] = datetime.now().isoformat()
        state.setdefault("errors", [])
        state.setdefault("warnings", [])
        logger.info("Starting contract processing pipeline")
        try:
            final_state = self.graph.invoke(state)
            logger.info("Pipeline completed successfully")
            return final_state
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

_orchestrator: Optional[ContractOrchestrator] = None


def get_orchestrator() -> ContractOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ContractOrchestrator()
    return _orchestrator