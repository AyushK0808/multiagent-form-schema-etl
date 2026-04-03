from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ui.services import (
    FIELD_TYPES,
    build_schema_template,
    list_schemas,
    load_schema_by_name,
    rows_to_schema,
    save_schema_payload,
    schema_to_rows,
)
from ui.state import load_schema_into_state, push_ui_log
from ui.widgets import hero, render_ui_logs, schema_preview_html


def screen_schema_editor() -> None:
    hero("Add or Update Schemas", "Screen 2")
    st.write(
        "Build schemas field-by-field in the **Form Builder** or paste raw JSON in the "
        "**JSON Editor**. Both views stay in sync."
    )

    labels, mapping = _schema_selector_options()
    selected_label = st.selectbox("Load existing schema or start fresh", labels, index=0)
    selected_schema_meta = mapping[selected_label]

    if selected_schema_meta is not None:
        loaded = load_schema_by_name(selected_schema_meta["form_name"])
        if loaded:
            load_schema_into_state(loaded)
            push_ui_log(
                "Loaded schema into editor",
                details={"form_name": loaded["form_name"], "schema_id": loaded.get("schema_id")},
            )

    st.divider()
    tab_form, tab_json = st.tabs(["Form Builder", "JSON Editor"])

    with tab_form:
        _render_form_builder()

    with tab_json:
        _render_json_editor(selected_schema_meta)

    render_ui_logs()


def _schema_selector_options():
    schemas = list_schemas()
    labels = ["-- Create New --"]
    mapping = {"-- Create New --": None}
    for schema in schemas:
        label = f'{schema["form_name"]}  ({schema["schema_id"][:8]}...)'
        labels.append(label)
        mapping[label] = schema
    return labels, mapping


def _render_form_builder() -> None:
    st.markdown("##### Schema Metadata")
    meta_c1, meta_c2, meta_c3 = st.columns([2, 1, 3])
    with meta_c1:
        fb_form_name = st.text_input(
            "Form Name",
            value=st.session_state.get("fb_form_name", "NewSchema"),
            key="fb_form_name_input",
        )
    with meta_c2:
        fb_version = st.text_input(
            "Version",
            value=st.session_state.get("fb_version", "1.0"),
            key="fb_version_input",
        )
    with meta_c3:
        fb_description = st.text_input(
            "Description",
            value=st.session_state.get("fb_description", ""),
            placeholder="One-line description of this schema",
            key="fb_description_input",
        )

    st.markdown("##### Fields")
    st.caption(
        "Add rows with **+** in the bottom-left corner of the table. "
        "Delete a row by selecting it and pressing the trash icon."
    )

    initial_rows = st.session_state.get("fb_rows") or schema_to_rows(build_schema_template())
    df = pd.DataFrame(initial_rows, columns=["name", "type", "description", "required", "examples"])
    df["required"] = df["required"].astype(bool)
    df["examples"] = df["examples"].astype(str).replace("nan", "")

    edited_df = st.data_editor(
        df,
        column_config={
            "name": st.column_config.TextColumn(
                "Field Name", help="Python-identifier-style name, e.g. effective_date", width="medium"
            ),
            "type": st.column_config.SelectboxColumn("Type", options=FIELD_TYPES, width="small"),
            "description": st.column_config.TextColumn(
                "Description", help="Human-readable description for the LLM prompt", width="large"
            ),
            "required": st.column_config.CheckboxColumn("Req?", width="small"),
            "examples": st.column_config.TextColumn(
                "Examples (comma-sep)", help='e.g. "2024-01-01, 2025-06-30"', width="medium"
            ),
        },
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="fb_data_editor",
    )

    fb_schema = rows_to_schema(
        edited_df.to_dict("records"),
        form_name=fb_form_name,
        version=fb_version,
        description=fb_description,
    )

    st.markdown("##### JSON Preview")
    with st.container():
        st.markdown(schema_preview_html(fb_schema), unsafe_allow_html=True)
    with st.expander("Raw JSON", expanded=False):
        st.code(json.dumps(fb_schema, indent=2), language="json")

    st.divider()
    save_col, info_col = st.columns([0.25, 0.75])
    with save_col:
        if st.button("Save Schema", use_container_width=True, key="fb_save"):
            try:
                schema_id = save_schema_payload(
                    fb_schema,
                    schema_id=st.session_state.get("selected_schema_id"),
                )
                st.session_state["selected_schema_id"] = schema_id
                st.session_state["schema_editor_json"] = json.dumps(fb_schema, indent=2)
                push_ui_log(
                    "Saved schema from form builder",
                    details={"form_name": fb_schema.get("form_name"), "schema_id": schema_id},
                )
                st.success(f"Saved - ID: `{schema_id}`", icon="✅")
            except Exception as exc:
                push_ui_log("Schema save failed", level="error", details={"error": str(exc)})
                st.error(str(exc))
    with info_col:
        st.caption(
            "Saving registers the schema in SQLite and the embedding registry. "
            "It will be available for pipeline runs immediately."
        )


def _render_json_editor(selected_schema_meta) -> None:
    if selected_schema_meta is None:
        name_hint = st.text_input(
            "New form name",
            value=st.session_state.get("generated_schema_name", "NewSchema"),
            key="json_new_form_name",
        )
        if st.button("Create Blank Template", key="json_create_blank"):
            template = build_schema_template(name_hint or "NewSchema")
            load_schema_into_state(template)
            push_ui_log("Created blank schema template", details={"form_name": template["form_name"]})
            st.rerun()

    editor_value = st.text_area(
        "Schema JSON",
        value=st.session_state["schema_editor_json"],
        height=520,
        key="json_editor_area",
    )
    st.session_state["schema_editor_json"] = editor_value

    json_preview_ok = False
    try:
        parsed = json.loads(editor_value)
        json_preview_ok = True
        st.markdown("##### Preview")
        st.markdown(schema_preview_html(parsed), unsafe_allow_html=True)
    except json.JSONDecodeError as err:
        st.warning(f"Invalid JSON - {err}", icon="⚠️")

    st.divider()
    json_save_col, sync_col, json_info_col = st.columns([0.2, 0.2, 0.6])
    with json_save_col:
        if st.button("Save Schema", use_container_width=True, key="json_save", disabled=not json_preview_ok):
            try:
                payload = json.loads(editor_value)
                schema_id = save_schema_payload(
                    payload,
                    schema_id=st.session_state.get("selected_schema_id"),
                )
                st.session_state["selected_schema_id"] = schema_id
                push_ui_log(
                    "Saved schema from JSON editor",
                    details={"form_name": payload.get("form_name"), "schema_id": schema_id},
                )
                st.success(f"Saved - ID: `{schema_id}`", icon="✅")
            except Exception as exc:
                push_ui_log("JSON schema save failed", level="error", details={"error": str(exc)})
                st.error(str(exc))
    with sync_col:
        if st.button("Sync to Form Builder", use_container_width=True, disabled=not json_preview_ok):
            try:
                parsed = json.loads(editor_value)
                load_schema_into_state(parsed)
                push_ui_log("Synced JSON editor into form builder", details={"form_name": parsed.get("form_name")})
                st.success("Synced to Form Builder.", icon="✅")
                st.rerun()
            except Exception as exc:
                push_ui_log("JSON sync failed", level="error", details={"error": str(exc)})
                st.error(str(exc))
    with json_info_col:
        st.caption("Required keys: `form_name`, `version`, `description`, `fields`.")
