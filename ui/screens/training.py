from __future__ import annotations

import time
from typing import List

import pandas as pd
import streamlit as st

from finetune.adapter_groups import ADAPTER_GROUPS
from finetune.config import DATASET_SPECS
from ui.services import (
    TrainingProcessMonitor,
    build_training_command,
    expected_training_targets,
    get_dataset_info,
    infer_current_stage,
    summarize_target_progress,
)
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

    _render_run_summary(
        mode=mode_key,
        model=model,
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        max_length=int(max_length),
        use_augmentation=use_augmentation,
        use_all_datasets=use_all_datasets,
        curriculum=curriculum,
        normal_datasets=normal_datasets,
        lora_groups=lora_groups,
        lora_datasets=lora_datasets,
    )

    st.markdown("##### Command Preview")
    st.code(" ".join(command), language="bash")
    dashboard = st.container()

    if st.button("Run Training", use_container_width=True):
        dashboard.markdown("#### Live Training Dashboard")
        push_ui_log(
            "Starting training command",
            details={"mode": mode_key, "model": model, "command": " ".join(command)},
        )
        monitor = TrainingProcessMonitor(command).start()
        targets = expected_training_targets(mode_key, model, lora_groups=lora_groups)

        status_placeholder = dashboard.empty()
        progress_placeholder = dashboard.empty()
        target_placeholder = dashboard.container()
        plots_placeholder = dashboard.container()
        log_placeholder = dashboard.empty()
        stderr_placeholder = dashboard.empty()

        returncode = None
        while returncode is None:
            returncode = monitor.poll()
            _render_live_training_dashboard(
                status_placeholder=status_placeholder,
                progress_placeholder=progress_placeholder,
                target_placeholder=target_placeholder,
                plots_placeholder=plots_placeholder,
                log_placeholder=log_placeholder,
                stderr_placeholder=stderr_placeholder,
                targets=targets,
                epochs=int(epochs),
                current_stage=infer_current_stage(monitor.combined_tail(120)),
                log_text=monitor.stdout_text(),
                stderr_text=monitor.stderr_text(),
            )
            time.sleep(0.5)

        returncode = monitor.wait()
        _render_live_training_dashboard(
            status_placeholder=status_placeholder,
            progress_placeholder=progress_placeholder,
            target_placeholder=target_placeholder,
            plots_placeholder=plots_placeholder,
            log_placeholder=log_placeholder,
            stderr_placeholder=stderr_placeholder,
            targets=targets,
            epochs=int(epochs),
            current_stage="Run finished" if returncode == 0 else "Run failed",
            log_text=monitor.stdout_text(),
            stderr_text=monitor.stderr_text(),
            final_state=True,
        )

        if returncode == 0:
            push_ui_log("Training command completed successfully")
            st.success("Training finished successfully.")
        else:
            push_ui_log(
                "Training command failed",
                level="error",
                details={"returncode": returncode},
            )
            st.error(f"Training failed with exit code {returncode}.")

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
                        f"Include {dataset_name}",
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
        st.warning("No datasets selected - training will be skipped.")
    else:
        st.success(
            f"{len(lora_datasets)} dataset(s) selected: " + ", ".join(f"`{name}`" for name in lora_datasets),
        )
    return lora_groups, lora_datasets


def _render_run_summary(
    mode: str,
    model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    use_augmentation: bool,
    use_all_datasets: bool,
    curriculum: bool,
    normal_datasets: List[str],
    lora_groups: List[str],
    lora_datasets: List[str],
) -> None:
    st.markdown("##### Run Summary")
    targets = expected_training_targets(mode, model, lora_groups=lora_groups)
    if mode == "full":
        dataset_count = len(DATASET_SPECS) if use_all_datasets else len(normal_datasets)
        mode_caption = f"Curriculum `{curriculum}`"
    else:
        dataset_count = len(lora_datasets)
        mode_caption = f"Groups `{', '.join(lora_groups) or 'none'}`"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Targets", len(targets))
    col2.metric("Epochs", epochs)
    col3.metric("Datasets", dataset_count)
    col4.metric("Max Length", max_length)
    st.caption(
        f"Batch size `{batch_size}` | Learning rate `{learning_rate:.6g}` | "
        f"Augmentation `{use_augmentation}` | {mode_caption}"
    )


def _render_live_training_dashboard(
    status_placeholder,
    progress_placeholder,
    target_placeholder,
    plots_placeholder,
    log_placeholder,
    stderr_placeholder,
    targets: List[dict],
    epochs: int,
    current_stage: str,
    log_text: str,
    stderr_text: str,
    final_state: bool = False,
) -> None:
    summaries = [summarize_target_progress(target, epochs) for target in targets]
    total_epochs = max(len(summaries) * max(epochs, 1), 1)
    finished_epochs = sum(summary["current_epoch"] for summary in summaries)
    finished_targets = sum(1 for summary in summaries if summary["done"])
    overall_progress = min((finished_epochs + finished_targets) / total_epochs, 1.0)

    if final_state:
        if stderr_text.strip():
            status_placeholder.error(f"Current stage: {current_stage}")
        else:
            status_placeholder.success(f"Current stage: {current_stage}")
    else:
        status_placeholder.info(f"Current stage: {current_stage}")

    progress_placeholder.progress(
        overall_progress,
        text=f"Overall progress {overall_progress * 100:.0f}% | {finished_targets}/{len(summaries)} targets complete",
    )

    with target_placeholder:
        target_placeholder.empty()
        st.markdown("##### Progress by Target")
        if not summaries:
            st.caption("No targets resolved for the current configuration.")
        for summary in summaries:
            st.markdown(f"**{summary['label']}**")
            metric_text = (
                f"{summary['primary_metric']}={summary['latest_metric']:.4f}"
                if isinstance(summary["latest_metric"], (float, int))
                else "metric pending"
            )
            st.progress(
                min(summary["progress"], 1.0),
                text=f"Epoch {summary['current_epoch']}/{epochs} | {metric_text}",
            )
            if summary["csv_rows"]:
                df = pd.DataFrame(summary["csv_rows"])
                chart_cols = [col for col in ("train_loss", "eval_loss", summary["primary_metric"]) if col in df.columns]
                if "epoch" in df.columns and chart_cols:
                    st.line_chart(df[["epoch", *chart_cols]].set_index("epoch"), height=180, use_container_width=True)
            if summary["metrics"]:
                st.json(summary["metrics"], expanded=False)

    with plots_placeholder:
        plots_placeholder.empty()
        st.markdown("##### Artifact Previews")
        shown = False
        for summary in summaries:
            overview_path = summary["plots_dir"] / "overview.png"
            if overview_path.exists():
                shown = True
                st.image(str(overview_path), caption=summary["label"], use_container_width=True)
        if not shown:
            st.caption("Plot previews will appear after the first epoch finishes and the trainer writes them.")

    log_placeholder.markdown("##### Live stdout")
    log_placeholder.code(log_text or "No stdout yet.", language="text")
    if stderr_text.strip():
        stderr_placeholder.markdown("##### Live stderr")
        stderr_placeholder.code(stderr_text, language="text")
    else:
        stderr_placeholder.caption("No stderr output.")
