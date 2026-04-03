from __future__ import annotations

from typing import List

import streamlit as st

from finetune.adapter_groups import ADAPTER_GROUPS
from finetune.config import DATASET_SPECS
from ui.logging_bridge import run_with_live_logs
from ui.services import build_training_command, get_dataset_info, run_training_command
from ui.state import push_ui_log
from ui.widgets import dataset_badge, hero, render_ui_logs


def screen_training() -> None:
    hero("Launch Fine-Tuning", "Screen 3")
    st.write("Configure and launch fine-tuning or LoRA adapter training.")

    mode = st.radio("Training mode", ["Full Fine-Tune", "LoRA Adapters"], horizontal=True)
    mode_key = "full" if mode == "Full Fine-Tune" else "lora"

    st.divider()
    st.markdown("##### Hyperparameters")
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
        default_lr = 3e-5 if mode_key == "full" else 3e-4
        learning_rate = st.number_input(
            "Learning rate",
            min_value=1e-6,
            max_value=1e-2,
            value=default_lr,
            format="%.6f",
        )
    with col6:
        use_augmentation = st.toggle("Enable augmentation", value=True)

    st.divider()
    use_all_datasets = False
    curriculum = False
    normal_datasets: List[str] = []
    lora_groups: List[str] = []
    lora_datasets: List[str] = []

    if mode_key == "full":
        st.markdown("##### Datasets")
        use_all_datasets = st.toggle("Use all datasets", value=False)
        if use_all_datasets:
            st.caption(f"All {len(DATASET_SPECS)} datasets from `DATASET_SPECS` will be used.")
        else:
            normal_datasets = st.multiselect(
                "Select datasets",
                list(DATASET_SPECS.keys()),
                default=["CORD", "SROIE"],
            )
            if normal_datasets:
                with st.expander("Dataset details", expanded=False):
                    for dataset_name in normal_datasets:
                        st.markdown(dataset_badge(dataset_name, get_dataset_info(dataset_name)), unsafe_allow_html=True)
        curriculum = st.toggle("Enable curriculum ordering", value=True)
    else:
        lora_groups, lora_datasets = _render_lora_dataset_picker()

    st.divider()
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
        lora_datasets=lora_datasets if lora_datasets else None,
    )

    st.markdown("##### Command Preview")
    st.code(" ".join(command), language="bash")
    trace_section = st.container()

    if st.button("Run Training", use_container_width=True):
        trace_section.markdown("#### Training Trace")
        trace_status = trace_section.empty()
        trace_output = trace_section.empty()
        push_ui_log(
            "Starting training command",
            details={"mode": mode_key, "model": model, "command": " ".join(command)},
        )
        with st.spinner("Training is running - this may take a while..."):
            result = run_with_live_logs(
                run_training_command,
                command,
                log_placeholder=trace_output,
                status_placeholder=trace_status,
            )

        if result["returncode"] == 0:
            push_ui_log("Training command completed successfully")
            st.success("Training finished successfully.", icon="✅")
            trace_status.success("Training trace complete.")
        else:
            push_ui_log(
                "Training command failed",
                level="error",
                details={"returncode": result["returncode"]},
            )
            st.error(f"Training failed with exit code {result['returncode']}.", icon="❌")
            trace_status.error("Training trace ended with errors.")

        if result["stdout"]:
            st.text_area("stdout", result["stdout"], height=260)
        if result["stderr"]:
            st.text_area("stderr", result["stderr"], height=220)

    render_ui_logs()


def _render_lora_dataset_picker() -> tuple[List[str], List[str]]:
    st.markdown("##### Adapter Groups & Datasets")
    group_names = [group.name for group in ADAPTER_GROUPS]
    lora_groups = st.multiselect(
        "Adapter groups to train",
        group_names,
        default=group_names,
    )

    lora_datasets: List[str] = []
    if not lora_groups:
        return lora_groups, lora_datasets

    group_map = {group.name: group for group in ADAPTER_GROUPS}
    st.markdown("**Select datasets to include (per-group filter):**")
    st.caption(
        "Unchecked datasets will be skipped even if they belong to a selected group. "
        "Useful for excluding large or gated datasets."
    )

    selected_datasets: List[str] = []
    for group_name in lora_groups:
        group = group_map.get(group_name)
        if not group:
            continue
        with st.expander(f"**{group_name}** - {group.label}", expanded=True):
            for dataset_name in group.datasets:
                if dataset_name not in DATASET_SPECS:
                    st.caption(f"Warning: `{dataset_name}` not in `DATASET_SPECS` - will be skipped.")
                    continue
                info = get_dataset_info(dataset_name)
                col_cb, col_info = st.columns([0.08, 0.92])
                with col_cb:
                    checked = st.checkbox(
                        "",
                        value=True,
                        key=f"lora_ds_{group_name}_{dataset_name}",
                        label_visibility="collapsed",
                    )
                with col_info:
                    st.markdown(dataset_badge(dataset_name, info), unsafe_allow_html=True)
                if checked:
                    selected_datasets.append(dataset_name)

    lora_datasets = list(dict.fromkeys(selected_datasets))
    if not lora_datasets:
        st.warning("No datasets selected - training will be skipped.", icon="⚠️")
    else:
        st.success(
            f"{len(lora_datasets)} dataset(s) selected: " + ", ".join(f"`{name}`" for name in lora_datasets),
            icon="✅",
        )
    return lora_groups, lora_datasets
