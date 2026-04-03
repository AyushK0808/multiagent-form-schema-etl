from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st

from ui.services import build_schema_template, schema_to_rows


def bootstrap_state() -> None:
    st.session_state.setdefault(
        "schema_editor_json",
        json.dumps(build_schema_template(), indent=2),
    )
    st.session_state.setdefault("selected_schema_id", None)
    st.session_state.setdefault("generated_schema_name", "")
    template = build_schema_template()
    st.session_state.setdefault("fb_rows", schema_to_rows(template))
    st.session_state.setdefault("fb_form_name", "NewSchema")
    st.session_state.setdefault("fb_version", "1.0")
    st.session_state.setdefault("fb_description", "")
    st.session_state.setdefault("ui_event_log", [])


def load_schema_into_state(schema: Dict[str, Any]) -> None:
    st.session_state["schema_editor_json"] = json.dumps(schema, indent=2)
    st.session_state["fb_rows"] = schema_to_rows(schema)
    st.session_state["fb_form_name"] = schema.get("form_name", "")
    st.session_state["fb_version"] = schema.get("version", "1.0")
    st.session_state["fb_description"] = schema.get("description", "")
    st.session_state["selected_schema_id"] = schema.get("schema_id")


def push_ui_log(message: str, level: str = "info", details: Dict[str, Any] | None = None) -> None:
    event = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "level": level.upper(),
        "message": message,
    }
    if details:
        event["details"] = details
    st.session_state.setdefault("ui_event_log", []).append(event)
    st.session_state["ui_event_log"] = st.session_state["ui_event_log"][-100:]


def get_ui_logs() -> List[Dict[str, Any]]:
    return list(st.session_state.get("ui_event_log", []))
