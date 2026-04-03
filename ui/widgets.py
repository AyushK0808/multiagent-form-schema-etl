from __future__ import annotations

import json
from typing import Any, Dict

import streamlit as st

from ui.state import get_ui_logs


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="label">{subtitle}</div>'
        f'<div class="metric">{title}</div></div>',
        unsafe_allow_html=True,
    )


def schema_preview_html(schema: Dict[str, Any]) -> str:
    fields = schema.get("fields", {})
    if not fields:
        return "<p style='color:#888;font-size:0.85rem;'>No fields defined.</p>"
    rows_html = ""
    for field_name, field_meta in fields.items():
        field_type = field_meta.get("type", "string")
        description = field_meta.get("description", "") or "-"
        required = field_meta.get("required", False)
        badge = '<span class="badge-req">required</span>' if required else '<span class="badge-opt">optional</span>'
        examples = field_meta.get("examples", [])
        example_text = ", ".join(str(item) for item in examples[:3]) if examples else "-"
        rows_html += (
            f"<tr><td><code>{field_name}</code></td>"
            f"<td><code style='color:#ff9a3d'>{field_type}</code></td>"
            f"<td>{description}</td>"
            f"<td>{badge}</td>"
            f"<td style='color:#888;font-size:0.8rem'>{example_text}</td></tr>"
        )
    return (
        "<table class='schema-preview-table'>"
        "<thead><tr><th>Field</th><th>Type</th><th>Description</th>"
        "<th>Required</th><th>Examples</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


def dataset_badge(name: str, info: Dict[str, Any]) -> str:
    gated_badge = '<span class="badge-gated">gated</span>' if info.get("gated") else ""
    return (
        f"<div class='dataset-card'>"
        f"<strong>{name}</strong>{gated_badge}"
        f"<br/><span style='color:#aaa;font-size:0.82rem'>{info['description']}</span>"
        f"<br/><span style='color:#888;font-size:0.78rem'>"
        f"Size: <code>{info['size']}</code> &nbsp;|&nbsp; "
        f"Train: <code>{info['train_samples']}</code> &nbsp;|&nbsp; "
        f"Val: <code>{info['val_samples']}</code>"
        f"</span></div>"
    )


def render_ui_logs() -> None:
    logs = get_ui_logs()
    with st.expander("UI Event Log", expanded=False):
        if not logs:
            st.caption("No UI events recorded yet.")
            return
        st.code(json.dumps(logs[-25:], indent=2), language="json")

