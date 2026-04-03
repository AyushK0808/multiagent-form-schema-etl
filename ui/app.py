from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ui.screens import screen_recognition, screen_schema_editor, screen_training
from ui.logging_bridge import setup_ui_log_capture
from ui.state import bootstrap_state
from ui.theme import inject_theme


st.set_page_config(
    page_title="Schema ETL Console",
    page_icon=":orange_book:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    setup_ui_log_capture()
    inject_theme()
    bootstrap_state()

    with st.sidebar:
        st.markdown("## Schema ETL Console")
        screen = st.radio(
            "Navigate",
            [
                "Generate Schema",
                "Add / Update Schema",
                "Training",
            ],
        )
        st.divider()
        st.caption("Theme: orange + black")
        st.caption("UI actions are recorded in the event log on each screen.")

    if screen == "Generate Schema":
        screen_recognition()
    elif screen == "Add / Update Schema":
        screen_schema_editor()
    else:
        screen_training()


if __name__ == "__main__":
    main()
