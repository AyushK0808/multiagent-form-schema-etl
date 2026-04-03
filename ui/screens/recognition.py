from __future__ import annotations

import json

import streamlit as st

from ui.logging_bridge import run_with_live_logs
from ui.services import (
    list_schemas,
    load_preview_image,
    load_schema_by_name,
    recognize_schema_from_image,
    run_document_pipeline,
    save_uploaded_file,
)
from ui.state import load_schema_into_state, push_ui_log
from ui.widgets import hero, render_ui_logs, schema_preview_html


def screen_recognition() -> None:
    hero("Generate Schema From PDF or Image", "Screen 1")
    st.write(
        "Upload a document, preview the first page, and either use a stored schema "
        "directly or run the schema recognizer when no schema is selected."
    )

    schema_rows = list_schemas()
    none_option = "-- none --"
    schema_options = [none_option] + [row["form_name"] for row in schema_rows]
    selected_form_name = st.selectbox(
        "Select an existing schema (optional)",
        schema_options,
        help="If you pick a stored schema, the UI will use it directly and skip schema recognition.",
    )
    if selected_form_name != none_option:
        loaded = load_schema_by_name(selected_form_name)
        if loaded is not None:
            load_schema_into_state(loaded)
            st.session_state["generated_schema_name"] = loaded["form_name"]
            push_ui_log(
                "Selected stored schema",
                details={"form_name": loaded["form_name"], "schema_id": loaded.get("schema_id")},
            )
            st.caption(f"Selected `{loaded['form_name']}` from SQLite. Recognition will be skipped.")

    st.divider()

    uploaded = st.file_uploader(
        "Upload a document",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        help="PDFs render from page 1. Images are used directly.",
    )
    if not uploaded:
        st.info("Upload a PDF or image to begin.", icon="📄")
        render_ui_logs()
        return

    try:
        saved_path = save_uploaded_file(uploaded)
        preview = load_preview_image(saved_path)
        push_ui_log(
            "Prepared upload preview",
            details={"file_name": uploaded.name, "saved_path": str(saved_path), "suffix": saved_path.suffix.lower()},
        )
    except Exception as exc:
        push_ui_log("Failed to prepare upload preview", level="error", details={"error": str(exc)})
        st.error(f"Could not load file: {exc}")
        render_ui_logs()
        return

    trace_section = st.container()

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.image(preview, caption=f"Page 1 - {saved_path.name}", use_container_width=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"**File:** `{saved_path.name}`")

        if selected_form_name != none_option:
            st.caption("Using the selected schema directly, then running the pipeline.")
            action_label = "Use Selected Schema and Run Pipeline"
        else:
            st.caption("No schema selected. The recognition models will infer one, then run the pipeline.")
            action_label = "Recognize Schema and Run Pipeline"

        run_btn = st.button(action_label, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if run_btn:
            trace_section.markdown("#### Execution Trace")
            trace_status = trace_section.empty()
            trace_output = trace_section.empty()
            schema_payload, prediction, schema_found = run_with_live_logs(
                _resolve_schema,
                selected_form_name,
                none_option,
                preview,
                log_placeholder=trace_output,
                status_placeholder=trace_status,
            )

            if schema_payload is None or prediction is None:
                render_ui_logs()
                return

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Schema", prediction["schema_name"])
            m2.metric("Confidence", f'{prediction["confidence"]:.1%}')
            m3.metric("Source", prediction["source"].replace("_", " "))

            if selected_form_name != none_option:
                st.success("Selected schema loaded from SQLite. Recognition was skipped.", icon="✅")
            elif schema_found:
                st.success("Existing schema loaded from SQLite.", icon="✅")
            else:
                st.warning("No stored schema matched the prediction - blank template prepared.", icon="⚠️")

            load_schema_into_state(schema_payload)
            st.session_state["generated_schema_name"] = prediction["schema_name"]

            try:
                push_ui_log(
                    "Starting ETL pipeline",
                    details={
                        "file_path": str(saved_path),
                        "form_name": schema_payload.get("form_name"),
                        "schema_id": schema_payload.get("schema_id"),
                    },
                )
                with st.spinner("Running ETL pipeline..."):
                    pipeline_result = run_with_live_logs(
                        run_document_pipeline,
                        saved_path,
                        form_name=schema_payload.get("form_name"),
                        schema_id=schema_payload.get("schema_id"),
                        log_placeholder=trace_output,
                        status_placeholder=trace_status,
                    )
                push_ui_log(
                    "ETL pipeline finished",
                    details={"output_path": str(pipeline_result["output_path"])},
                )
                trace_status.success("Execution trace complete.")
            except Exception as exc:
                push_ui_log("ETL pipeline failed", level="error", details={"error": str(exc)})
                st.error(f"Pipeline execution failed: {exc}")
                if saved_path.suffix.lower() != ".pdf":
                    st.info("Pipeline execution is currently available only for PDF uploads.", icon="💡")
                render_ui_logs()
                return

            _render_pipeline_output(pipeline_result)
            _render_schema_preview(schema_payload)
            st.info("Switch to **Add / Update Schema** to edit fields or save this schema.", icon="✏️")

    render_ui_logs()


def _resolve_schema(selected_form_name: str, none_option: str, preview):
    if selected_form_name != none_option:
        schema_payload = load_schema_by_name(selected_form_name)
        if schema_payload is None:
            push_ui_log("Stored schema lookup failed", level="error", details={"form_name": selected_form_name})
            st.error(f"Stored schema '{selected_form_name}' could not be loaded.")
            return None, None, False
        prediction = {
            "schema_name": schema_payload.get("form_name", selected_form_name),
            "confidence": 1.0,
            "source": "manual_selection",
        }
        push_ui_log(
            "Using manually selected schema",
            details={"form_name": prediction["schema_name"], "schema_id": schema_payload.get("schema_id")},
        )
        return schema_payload, prediction, True

    with st.spinner("Recognizing schema..."):
        try:
            result = recognize_schema_from_image(preview)
        except Exception as exc:
            push_ui_log("Schema recognition failed", level="error", details={"error": str(exc)})
            st.error(f"Recognition failed: {exc}")
            st.info(
                "Tip: make sure Tesseract is installed, or that HuggingFace fallback models are reachable.",
                icon="💡",
            )
            return None, None, False

    prediction = result["prediction"]
    schema_payload = result["schema_payload"]
    push_ui_log(
        "Schema recognized",
        details={
            "schema_name": prediction["schema_name"],
            "confidence": prediction["confidence"],
            "source": prediction["source"],
            "schema_found": result["schema_found"],
        },
    )
    return schema_payload, prediction, result["schema_found"]


def _render_pipeline_output(pipeline_result) -> None:
    pipeline_output = pipeline_result["output"]
    pipeline_meta = pipeline_output.get("pipeline_metadata", {})

    st.markdown("#### Pipeline Output")
    p1, p2, p3 = st.columns(3)
    p1.metric("Record ID", str(pipeline_output.get("record_id", "-"))[:12] or "-")
    p2.metric("Complete", "Yes" if pipeline_output.get("is_complete") else "No")
    p3.metric("Coverage", f'{pipeline_meta.get("field_coverage", 0.0):.1%}')
    st.caption(f"Saved output to `{pipeline_result['output_path']}`")

    fields = pipeline_output.get("fields", {})
    if fields:
        with st.expander("Extracted Fields", expanded=True):
            st.json(fields)

    warnings = pipeline_meta.get("warnings", [])
    errors = pipeline_meta.get("errors", [])
    if warnings:
        st.warning("Pipeline warnings:\n" + "\n".join(f"- {warning}" for warning in warnings[:5]))
    if errors:
        st.error("Pipeline errors:\n" + "\n".join(f"- {error}" for error in errors[:5]))


def _render_schema_preview(schema_payload) -> None:
    st.markdown("#### Schema Preview")
    with st.container():
        meta_col, _ = st.columns([2, 1])
        with meta_col:
            st.markdown(
                f"**Form:** `{schema_payload.get('form_name', '-')}`  "
                f"&nbsp;&nbsp; **Version:** `{schema_payload.get('version', '1.0')}`"
            )
            if schema_payload.get("description"):
                st.caption(schema_payload["description"])
        st.markdown(
            schema_preview_html(schema_payload),
            unsafe_allow_html=True,
        )

    with st.expander("Raw JSON", expanded=False):
        st.code(json.dumps(schema_payload, indent=2), language="json")
