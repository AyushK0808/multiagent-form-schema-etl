"""
Agentic ETL Orchestrator
========================
LangGraph pipeline with eight sequential nodes:

  layout          -> Layout-aware clause extraction (LoRA adapter / heuristic)
  parallel_extract -> Donut + LayoutLM run concurrently; produces ExtractionBundle
  policy_fuse     -> ReflexivePolicyLayer fuses both sets of candidates
  repair          -> Groq repair pass on fused fields
  schema_resolve  -> Groq agent normalises, searches registry, maps or synthesises
  validate        -> Field validation + recovery
  db_populate     -> Inserts mapped record into SQLite
  finalize        -> Packages final output dict

LoRA adapter routing
--------------------
The adapter group is resolved once from the recognised schema name at the
start of the pipeline (via adapter_groups.group_for_schema) and stored in
state["adapter_group"].  Both the layout node and parallel_extract node
read this value to activate the correct LoRA adapter.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph

from config.config import get_config
from layout_analysis.layout_structure import LayoutAnalyzer
from utils.validation import ValidationRecoveryManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ContractState(TypedDict, total=False):
    # Inputs
    blocks:       list
    page_image:   Any
    pdf_metadata: dict

    # LoRA routing — resolved once from schema_recognition.form_name
    adapter_group: str   # "group_1" | "group_2" | "group_3"

    # Layout
    clause_graph:       dict
    layout_predictions: list

    # Parallel extraction
    bundle: Any   # ExtractionBundle

    # Policy fusion
    policy_result: Any   # PolicyResult

    # Schema resolution
    schema:            dict
    resolved_schema:   dict
    schema_id:         str
    field_mapping:     dict
    normalised_fields: dict
    repaired_fields:   dict

    # DB
    record_id: str

    # Legacy
    form: Any   # FormInstance

    # Output
    output: dict

    # Metadata
    pipeline_start: str
    pipeline_end:   str
    errors:         List[str]
    warnings:       List[str]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ContractOrchestrator:

    def __init__(self):
        self.config          = get_config()
        self.layout_analyzer = LayoutAnalyzer()
        self.validator       = ValidationRecoveryManager()
        self.graph           = self._build_graph()

    def _build_graph(self) -> StateGraph:
        g = StateGraph(ContractState)

        g.add_node("layout",           self._layout_node)
        g.add_node("parallel_extract", self._parallel_extract_node)
        g.add_node("policy_fuse",      self._policy_fuse_node)
        g.add_node("repair",           self._repair_node)
        g.add_node("schema_resolve",   self._schema_resolve_node)
        g.add_node("validate",         self._validation_node)
        g.add_node("db_populate",      self._db_populate_node)
        g.add_node("finalize",         self._finalize_node)

        g.set_entry_point("layout")
        g.add_edge("layout",           "parallel_extract")
        g.add_edge("parallel_extract", "policy_fuse")
        g.add_edge("policy_fuse",      "repair")
        g.add_edge("repair",           "schema_resolve")
        g.add_edge("schema_resolve",   "validate")
        g.add_edge("validate",         "db_populate")
        g.add_edge("db_populate",      "finalize")

        return g.compile()

    # ------------------------------------------------------------------
    # Adapter group resolution  (called before the graph runs)
    # ------------------------------------------------------------------

    def _resolve_adapter_group(self, state: ContractState) -> str:
        """
        Determine which LoRA adapter group to use for this document.

        Priority:
        1. state["adapter_group"]  — caller pre-set it
        2. schema_recognition.form_name  → look up in SCHEMA_TO_GROUP
        3. Default "group_2"
        """
        if state.get("adapter_group"):
            return state["adapter_group"]

        form_name = (
            state.get("schema_recognition", {}).get("form_name")
            or state.get("schema", {}).get("form_name", "")
        )
        if form_name:
            try:
                from finetune.adapter_groups import group_for_schema
                group = group_for_schema(form_name.lower())
                logger.info(
                    "[Orchestrator] Resolved adapter group '%s' for schema '%s'",
                    group, form_name,
                )
                return group
            except ImportError:
                pass

        return "group_2"

    # ------------------------------------------------------------------
    # Node 1 — Layout analysis
    # ------------------------------------------------------------------

    def _layout_node(self, state: ContractState) -> ContractState:
        logger.info("--- Node: layout ---")
        group = state.get("adapter_group", "group_2")
        try:
            result = self.layout_analyzer.analyze(
                state["blocks"],
                state["page_image"],
                group_name=group,
            )
            state["clause_graph"]       = result["clause_graph"]
            state["layout_predictions"] = result.get("predictions", [])
            logger.info(
                "[Layout] %d clauses (adapter_group=%s)",
                len(state["clause_graph"]), group,
            )
        except Exception as exc:
            logger.error("[Layout] Failed: %s", exc)
            state.setdefault("errors", []).append(f"Layout: {exc}")
            state["clause_graph"] = {}

        if not state.get("schema"):
            logger.warning("[Layout] No schema present in state")
            state["schema"] = {}

        return state

    # ------------------------------------------------------------------
    # Node 2 — Parallel extraction (Donut + LayoutLM)
    # ------------------------------------------------------------------

    def _parallel_extract_node(self, state: ContractState) -> ContractState:
        logger.info("--- Node: parallel_extract ---")
        group = state.get("adapter_group", "group_2")
        try:
            full_text = " ".join(str(b.get("text", "")) for b in state.get("blocks", []))

            if self.config.enable_parallel_extraction:
                from extraction.parallel_extractor import ParallelExtractor
                extractor = ParallelExtractor(
                    donut_model_id=self.config.model.donut_model,
                    adapter_group=group,
                )
                bundle = extractor.extract(
                    blocks=state.get("blocks", []),
                    page_image=state["page_image"],
                    schema=state["schema"],
                    clause_graph=state.get("clause_graph", {}),
                    full_text=full_text,
                )
            else:
                logger.info("[ParallelExtract] Donut disabled — LayoutLM only")
                from extraction.parallel_extractor import _run_layoutlm, ExtractionBundle
                lm_result = _run_layoutlm(
                    state.get("blocks", []),
                    state["page_image"],
                    state["schema"],
                    state.get("clause_graph", {}),
                    full_text,
                    adapter_group=group,
                )
                bundle = ExtractionBundle(donut=None, layoutlm=lm_result)

            state["bundle"] = bundle
            logger.info("[ParallelExtract] %s", bundle.summary())

        except Exception as exc:
            logger.error("[ParallelExtract] Failed: %s", exc)
            state.setdefault("errors", []).append(f"ParallelExtract: {exc}")
            from extraction.parallel_extractor import ExtractionBundle
            state["bundle"] = ExtractionBundle()

        return state

    # ------------------------------------------------------------------
    # Node 3 — Reflexive policy fusion
    # ------------------------------------------------------------------

    def _policy_fuse_node(self, state: ContractState) -> ContractState:
        logger.info("--- Node: policy_fuse ---")
        try:
            from extraction.policy_layer import ReflexivePolicyLayer
            layer  = ReflexivePolicyLayer(
                conf_floor=1.0 - self.config.processing.confidence_threshold
            )
            result = layer.fuse(state["bundle"], state["schema"])
            state["policy_result"] = result
            logger.info(
                "[Policy] coverage=%.2f consistent=%s issues=%d",
                result.coverage, result.consistency_ok, len(result.issues),
            )
            for issue in result.issues:
                state.setdefault("warnings", []).append(f"Policy: {issue}")
        except Exception as exc:
            logger.error("[Policy] Failed: %s", exc)
            state.setdefault("errors", []).append(f"Policy: {exc}")
            from extraction.policy_layer import PolicyResult
            lm = state["bundle"].layoutlm if state.get("bundle") else None
            state["policy_result"] = PolicyResult(
                fields=lm.fields if lm else {},
                confidences=lm.confidences if lm else {},
                spatial_meta=lm.spatial_meta if lm else {},
                coverage=lm.field_coverage if lm else 0.0,
                consistency_ok=False,
                issues=[str(exc)],
            )
        return state

    # ------------------------------------------------------------------
    # Node 4 — Groq repair pass
    # ------------------------------------------------------------------

    def _repair_node(self, state: ContractState) -> ContractState:
        logger.info("--- Node: repair ---")
        pr     = state.get("policy_result")
        fields = pr.fields if pr else {}
        schema = state.get("schema", {})

        if not self.config.enable_schema_agent or not fields or not schema:
            state["repaired_fields"] = fields
            return state

        try:
            from schema.schema_agent import SchemaAgent
            agent = SchemaAgent(
                small_model=self.config.groq.small_model,
                synthesis_model=self.config.groq.synthesis_model,
            )
            repaired = agent.repair_fields(
                fields=fields,
                schema=schema,
                document_hint=schema.get("form_name", ""),
            )
            state["repaired_fields"] = repaired
            logger.info("[Repair] Repaired %d field(s)", len(repaired))
        except Exception as exc:
            logger.error("[Repair] Failed: %s", exc)
            state.setdefault("warnings", []).append(f"Repair: {exc}")
            state["repaired_fields"] = fields

        return state

    # ------------------------------------------------------------------
    # Node 5 — Schema resolution
    # ------------------------------------------------------------------

    def _schema_resolve_node(self, state: ContractState) -> ContractState:
        logger.info("--- Node: schema_resolve ---")
        pr     = state.get("policy_result")
        fields = state.get("repaired_fields") or (pr.fields if pr else {})

        if not self.config.enable_schema_agent:
            state["resolved_schema"]   = state.get("schema", {})
            state["normalised_fields"] = fields
            state["field_mapping"]     = {k: k for k in fields}
            return state

        try:
            from schema.schema_registry import SchemaRegistry
            from schema.schema_agent    import SchemaAgent

            registry = SchemaRegistry(
                registry_dir=self.config.paths.registry_dir,
                sim_threshold=self.config.processing.schema_sim_threshold,
            )
            agent = SchemaAgent(
                small_model=self.config.groq.small_model,
                synthesis_model=self.config.groq.synthesis_model,
            )

            doc_hint    = state.get("schema", {}).get("form_name", "")
            norm_fields = agent.normalise_fields(fields, document_hint=doc_hint)
            state["normalised_fields"] = norm_fields

            hits = registry.find_similar(norm_fields, top_k=1)
            if hits:
                sim, schema_id, matched_schema = hits[0]
                logger.info(
                    "[SchemaResolve] Registry hit: '%s' (sim=%.3f)",
                    matched_schema.get("form_name"), sim,
                )
                mapped_values, mapping = agent.map_fields(norm_fields, matched_schema)
                state["resolved_schema"]   = matched_schema
                state["schema_id"]         = schema_id
                state["field_mapping"]     = mapping
                state["normalised_fields"] = mapped_values
            else:
                logger.info("[SchemaResolve] No match — synthesising new schema")
                new_schema = agent.synthesise_schema(norm_fields, document_hint=doc_hint)
                schema_id  = registry.register(new_schema)
                state["resolved_schema"]   = new_schema
                state["schema_id"]         = schema_id
                state["field_mapping"]     = {k: k for k in norm_fields}
                state.setdefault("warnings", []).append(
                    f"New schema synthesised: '{new_schema.get('form_name')}' ({schema_id})"
                )

        except Exception as exc:
            logger.error("[SchemaResolve] Failed: %s", exc)
            state.setdefault("errors", []).append(f"SchemaResolve: {exc}")
            state["resolved_schema"]   = state.get("schema", {})
            state["normalised_fields"] = fields
            state["field_mapping"]     = {k: k for k in fields}
            state.setdefault("schema_id", "")

        return state

    # ------------------------------------------------------------------
    # Node 6 — Validation + recovery
    # ------------------------------------------------------------------

    def _validation_node(self, state: ContractState) -> ContractState:
        logger.info("--- Node: validate ---")
        if not self.config.enable_validation:
            return state

        fields = state.get("normalised_fields", {})
        schema = state.get("resolved_schema", state.get("schema", {}))
        if not schema or not fields:
            return state

        try:
            recovered, remaining = self.validator.validate_and_recover(
                fields, schema, state.get("clause_graph", {}),
                page_image=state.get("page_image"),
            )
            state["normalised_fields"] = recovered
            if remaining:
                state.setdefault("errors", []).extend(remaining)
                logger.warning("[Validate] %d unrecovered errors", len(remaining))
            else:
                logger.info("[Validate] All fields valid")
        except Exception as exc:
            logger.error("[Validate] Failed: %s", exc)
            state.setdefault("errors", []).append(f"Validate: {exc}")

        return state

    # ------------------------------------------------------------------
    # Node 7 — Database population
    # ------------------------------------------------------------------

    def _db_populate_node(self, state: ContractState) -> ContractState:
        logger.info("--- Node: db_populate ---")
        if not self.config.enable_db_population:
            return state

        try:
            from database.db_manager import DatabaseManager
            db = DatabaseManager(db_url=str(self.config.paths.db_path))

            schema   = state.get("resolved_schema") or state.get("schema", {})
            fields   = state.get("normalised_fields", {})
            pr       = state.get("policy_result")
            conf_avg = (
                sum(pr.confidences.values()) / len(pr.confidences)
                if pr and pr.confidences else 0.0
            )
            pdf_meta = state.get("pdf_metadata", {})
            source   = pdf_meta.get("title") or pdf_meta.get("source_path", "")

            record_id = db.insert_record(
                schema=schema,
                fields=fields,
                schema_id=state.get("schema_id", ""),
                source_doc=source,
                confidence_avg=conf_avg,
            )
            state["record_id"] = record_id
            logger.info("[DB] Record inserted: %s", record_id)

        except Exception as exc:
            logger.error("[DB] Population failed: %s", exc)
            state.setdefault("errors", []).append(f"DB: {exc}")

        return state

    # ------------------------------------------------------------------
    # Node 8 — Finalize
    # ------------------------------------------------------------------

    def _finalize_node(self, state: ContractState) -> ContractState:
        logger.info("--- Node: finalize ---")
        pr     = state.get("policy_result")
        schema = state.get("resolved_schema") or state.get("schema", {})
        fields = state.get("normalised_fields", {})

        output: Dict[str, Any] = {
            "form":           schema.get("form_name", "Unknown"),
            "schema_id":      state.get("schema_id", ""),
            "record_id":      state.get("record_id", ""),
            "adapter_group":  state.get("adapter_group", ""),
            "fields":         fields,
            "field_mapping":  state.get("field_mapping", {}),
            "is_complete":    _check_complete(fields, schema),
            "pipeline_metadata": {
                "start_time":     state.get("pipeline_start"),
                "end_time":       datetime.now().isoformat(),
                "num_clauses":    len(state.get("clause_graph", {})),
                "field_coverage": pr.coverage if pr else 0.0,
                "consistency_ok": pr.consistency_ok if pr else True,
                "errors":         state.get("errors", []),
                "warnings":       state.get("warnings", []),
                "pdf_metadata":   state.get("pdf_metadata", {}),
                "confidences":    pr.confidences if pr else {},
                "spatial_meta":   pr.spatial_meta if pr else {},
            },
        }
        state["output"]       = output
        state["pipeline_end"] = datetime.now().isoformat()
        logger.info(
            "[Finalize] form=%s complete=%s record=%s group=%s",
            output["form"], output["is_complete"],
            output["record_id"], output["adapter_group"],
        )
        return state

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def process(self, state: ContractState) -> ContractState:
        state["pipeline_start"] = datetime.now().isoformat()
        state.setdefault("errors",   [])
        state.setdefault("warnings", [])

        # Resolve adapter group once before the graph runs
        state["adapter_group"] = self._resolve_adapter_group(state)
        logger.info(
            "[Orchestrator] adapter_group=%s  schema=%s",
            state["adapter_group"],
            state.get("schema", {}).get("form_name", "?"),
        )

        logger.info("=" * 55)
        logger.info("  Agentic ETL Fabric — starting pipeline")
        logger.info("=" * 55)
        final = self.graph.invoke(state)
        logger.info("=" * 55)
        logger.info("  Pipeline complete")
        logger.info("=" * 55)
        return final


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_complete(fields: Dict, schema: Dict) -> bool:
    for fname, meta in schema.get("fields", {}).items():
        if meta.get("required") and fields.get(fname) is None:
            return False
    return True


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_orchestrator: Optional[ContractOrchestrator] = None


def get_orchestrator() -> ContractOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ContractOrchestrator()
    return _orchestrator