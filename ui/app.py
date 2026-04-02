from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from finetune.adapter_groups import ADAPTER_GROUPS
from finetune.config import DATASET_SPECS
from ui.services import (
    build_schema_template,
    build_training_command,
    list_schemas,
    load_preview_image,
    load_schema_by_name,
    recognize_schema_from_image,
    run_training_command,
    save_schema_payload,
    save_uploaded_file,
)


st.set_page_config(
    page_title="Schema ETL Console",
    page_icon=":orange_book:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b0b0b;
            --panel: #141414;
            --panel-2: #1d1d1d;
            --border: #2f2f2f;
            --accent: #ff7a00;
            --accent-2: #ff9a3d;
            --text: #f5f5f5;
            --muted: #b7b7b7;
        }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(255,122,0,0.18), transparent 30%),
                linear-gradient(180deg, #090909 0%, #111111 100%);
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #120d08 0%, #0c0c0c 100%);
            border-right: 1px solid var(--border);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 1.5rem;
            border: 1px solid rgba(255,122,0,0.25);
            background: linear-gradient(135deg, rgba(255,122,0,0.18), rgba(20,20,20,0.92));
            border-radius: 20px;
            margin-bottom: 1rem;
            box-shadow: 0 18px 48px rgba(0,0,0,0.35);
        }
        .card {
            background: linear-gradient(180deg, rgba(22,22,22,0.96), rgba(14,14,14,0.96));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }
        .label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--accent-2);
        }
        .metric {
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 0.35rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(18,18,18,0.9);
            border: 1px solid rgba(255,122,0,0.15);
            padding: 0.75rem;
            border-radius: 16px;
        }
        .stButton > button,
        .stDownloadButton > button {
            background: linear-gradient(180deg, #ff8b1f 0%, #ff6a00 100%);
            color: #111111;
            border: none;
            font-weight: 700;
            border-radius: 999px;
        }
        .stTextArea textarea,
        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"],
        .stMultiSelect div[data-baseweb="select"] {
            background: rgba(18,18,18,0.95);
            color: var(--text);
        }
        pre, code {
            border-radius: 14px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def bootstrap_state() -> None:
    st.session_state.setdefault("schema_editor_json", json.dumps(build_schema_template(), indent=2))
    st.session_state.setdefault("selected_schema_id", None)
    st.session_state.setdefault("generated_schema_name", "")


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="label">{subtitle}</div>
            <div class="metric">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def screen_recognition() -> None:
    hero("Generate Schema From PDF or Image", "Screen 1")
    st.write("Upload a PDF or image, preview the first page, and run the current schema recognizer.")

    schema_rows = list_schemas()
    schema_options = ["None"] + [row["form_name"] for row in schema_rows]
    selected_form_name = st.selectbox(
        "Schema from SQLite",
        schema_options,
        help="Choose an existing stored schema to inspect or use as the target form.",
    )

    if selected_form_name != "None":
        selected_schema = load_schema_by_name(selected_form_name)
        if selected_schema is not None:
            st.session_state["schema_editor_json"] = json.dumps(selected_schema, indent=2)
            st.session_state["generated_schema_name"] = selected_schema["form_name"]
            st.session_state["selected_schema_id"] = selected_schema.get("schema_id")
            st.caption(f'Loaded `{selected_schema["form_name"]}` from SQLite.')

    uploaded = st.file_uploader(
        "Upload a document",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        help="PDFs are rendered from page 1. Images are used directly.",
    )

    if not uploaded:
        return

    saved_path = save_uploaded_file(uploaded)
    preview = load_preview_image(saved_path)

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.image(preview, caption=f"Preview: {saved_path.name}", use_container_width=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"Saved to `{saved_path}`")
        if st.button("Generate Schema Prediction", use_container_width=True):
            with st.spinner("Running schema recognition..."):
                try:
                    result = recognize_schema_from_image(preview)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    prediction = result["prediction"]
                    st.success("Prediction complete")
                    st.metric("Predicted Schema", prediction["schema_name"])
                    st.metric("Confidence", f'{prediction["confidence"]:.3f}')
                    st.caption(f'Source: `{prediction["source"]}`')

                    if selected_form_name != "None":
                        st.info(f"SQLite target selected: `{selected_form_name}`")

                    schema_payload = result["schema_payload"]
                    st.session_state["schema_editor_json"] = json.dumps(schema_payload, indent=2)
                    st.session_state["generated_schema_name"] = prediction["schema_name"]
                    st.session_state["selected_schema_id"] = schema_payload.get("schema_id")

                    if result["schema_found"]:
                        st.info("Existing schema found in SQLite and loaded into the editor.")
                    else:
                        st.warning("No stored schema matched the prediction. A blank template was prepared.")

                    st.code(json.dumps(schema_payload, indent=2), language="json")
        st.markdown("</div>", unsafe_allow_html=True)


def _schema_selector_options():
    schemas = list_schemas()
    labels = ["Create New"]
    mapping = {"Create New": None}
    for schema in schemas:
        label = f'{schema["form_name"]} ({schema["schema_id"][:8]})'
        labels.append(label)
        mapping[label] = schema
    return labels, mapping


def screen_schema_editor() -> None:
    hero("Add or Update Schemas", "Screen 2")
    st.write("Edit schema JSON directly and persist it to the SQLite schema store.")

    labels, mapping = _schema_selector_options()
    selected_label = st.selectbox("Load schema", labels, index=0)

    selected_schema = mapping[selected_label]
    if selected_schema is None:
        form_name_hint = st.text_input("New form name", value=st.session_state.get("generated_schema_name", "NewSchema"))
        if st.button("Create Blank Template"):
            template = build_schema_template(form_name_hint or "NewSchema")
            st.session_state["schema_editor_json"] = json.dumps(template, indent=2)
            st.session_state["selected_schema_id"] = None
    else:
        loaded = load_schema_by_name(selected_schema["form_name"])
        if loaded is not None:
            st.session_state["schema_editor_json"] = json.dumps(loaded, indent=2)
            st.session_state["selected_schema_id"] = loaded.get("schema_id")

    editor_value = st.text_area(
        "Schema JSON",
        value=st.session_state["schema_editor_json"],
        height=520,
    )
    st.session_state["schema_editor_json"] = editor_value

    col_a, col_b = st.columns([0.25, 0.75])
    with col_a:
        if st.button("Save Schema", use_container_width=True):
            try:
                payload = json.loads(editor_value)
                schema_id = save_schema_payload(payload, schema_id=st.session_state.get("selected_schema_id"))
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state["selected_schema_id"] = schema_id
                st.success(f"Schema saved as `{schema_id}`")
    with col_b:
        st.caption("Required shape: `form_name`, `version`, `description`, `fields`.")


def screen_training() -> None:
    hero("Launch Fine-Tuning", "Screen 3")
    st.write("Trigger the existing training scripts for either full fine-tuning or LoRA adapter training.")

    mode = st.radio("Training mode", ["Full Fine-Tune", "LoRA Adapters"], horizontal=True)
    mode_key = "full" if mode == "Full Fine-Tune" else "lora"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        model = st.selectbox("Model", ["both", "layoutlmv3", "donut"])
    with col2:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10, step=1)
    with col3:
        batch_size = st.number_input("Batch size", min_value=1, max_value=32, value=2, step=1)
    with col4:
        max_length = st.number_input("Max length", min_value=64, max_value=4096, value=512, step=64)

    col5, col6 = st.columns(2)
    with col5:
        learning_rate = st.number_input("Learning rate", min_value=0.000001, max_value=0.01, value=0.00003 if mode_key == "full" else 0.0003, format="%.6f")
    with col6:
        use_augmentation = st.toggle("Enable augmentation", value=True)

    use_all_datasets = False
    curriculum = False
    normal_datasets = []
    lora_groups = []

    if mode_key == "full":
        use_all_datasets = st.toggle("Use all datasets", value=False)
        if use_all_datasets:
            st.caption(f"All datasets from `finetune.config.DATASET_SPECS` will be used ({len(DATASET_SPECS)} total).")
        else:
            normal_datasets = st.multiselect("Datasets", list(DATASET_SPECS.keys()), default=["CORD", "SROIE"])
        curriculum = st.toggle("Enable curriculum ordering", value=True)
    else:
        group_names = [group.name for group in ADAPTER_GROUPS]
        lora_groups = st.multiselect("Adapter groups", group_names, default=group_names)

    command = build_training_command(
        mode=mode_key,
        model=model,
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        max_length=int(max_length),
        use_augmentation=use_augmentation,
        normal_datasets=normal_datasets,
        use_all_datasets=use_all_datasets,
        curriculum=curriculum,
        lora_groups=lora_groups,
    )

    st.code(" ".join(command), language="bash")

    if st.button("Run Training", use_container_width=True):
        with st.spinner("Training command is running. This may take a while..."):
            result = run_training_command(command)

        if result["returncode"] == 0:
            st.success("Training finished successfully.")
        else:
            st.error(f'Training failed with exit code {result["returncode"]}.')

        if result["stdout"]:
            st.text_area("stdout", result["stdout"], height=260)
        if result["stderr"]:
            st.text_area("stderr", result["stderr"], height=220)


def main() -> None:
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
        st.caption("Theme: orange + black")

    if screen == "Generate Schema":
        screen_recognition()
    elif screen == "Add / Update Schema":
        screen_schema_editor()
    else:
        screen_training()


if __name__ == "__main__":
    main()
