from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from empathic_common import (
    TASK_LABEL_NAME,
    TASK_LABEL_NUMERIC,
    discover_task_files,
    load_label_sheet,
    nasa_tlx_for_task,
    stress_class,
    stress_intervals_for_task,
    stress_value_for_window,
    subject_sort_key,
)


def label_cache_for_archives(features_df: pd.DataFrame) -> dict[str, pd.DataFrame | None]:
    cache: dict[str, pd.DataFrame | None] = {}
    for archive_text in sorted(features_df["archive_path"].dropna().unique()):
        archive = Path(archive_text)
        try:
            task_map = discover_task_files(archive)
            label_member = next((tf.label_path for tf in task_map.values() if tf.label_path), None)
            cache[archive_text] = load_label_sheet(archive, label_member) if label_member else None
        except Exception:
            cache[archive_text] = None
    return cache


def make_labels(features_path: Path, output_path: Path, excluded_subjects: set[str]) -> pd.DataFrame:
    features_df = pd.read_csv(features_path)
    if features_df.empty:
        raise ValueError(f"{features_path} is empty; run make_windows.py first")
    label_cache = label_cache_for_archives(features_df)

    labeled = features_df.copy()
    l_self_value = []
    l_self = []
    l_task = []
    l_task_name = []
    nasa_values = []
    is_labeled_analysis = []

    for row in labeled.itertuples(index=False):
        subject_id = row.subject_id
        task_id = row.task_id
        label_df = label_cache.get(row.archive_path)

        task_numeric = TASK_LABEL_NUMERIC.get(task_id, np.nan)
        l_task.append(task_numeric)
        l_task_name.append(TASK_LABEL_NAME.get(task_id, "unknown"))

        excluded = subject_id in excluded_subjects
        if excluded:
            stress_value = np.nan
            klass = np.nan
        else:
            intervals = stress_intervals_for_task(label_df, task_id)
            stress_value = stress_value_for_window(intervals, float(row.start_sec), float(row.end_sec))
            klass = stress_class(stress_value)
        l_self_value.append(stress_value if stress_value is not None else np.nan)
        l_self.append(klass)
        is_labeled_analysis.append(bool(not excluded and np.isfinite(klass)))

        nasa = nasa_tlx_for_task(label_df, task_id)
        nasa_values.append(nasa if nasa is not None else np.nan)

    labeled["L_self_value"] = l_self_value
    labeled["L_self"] = l_self
    labeled["L_task"] = l_task
    labeled["L_task_name"] = l_task_name
    labeled["L_nasa_continuous"] = nasa_values

    nasa_series = labeled["L_nasa_continuous"]
    if nasa_series.notna().sum() >= 3 and nasa_series.nunique(dropna=True) >= 3:
        labeled["L_nasa_tertile"] = pd.qcut(nasa_series, q=3, labels=[0, 1, 2], duplicates="drop").astype(float)
    else:
        labeled["L_nasa_tertile"] = np.nan
    labeled["is_labeled_analysis"] = is_labeled_analysis

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output_path, index=False)
    return labeled
