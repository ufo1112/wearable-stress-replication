from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_common import (  # noqa: E402
    BOOTSTRAP_N,
    RANDOM_STATE,
    RunLogger,
    event_equal_weights,
    robust_impute_scale,
    seed_everything,
    stable_seed,
    write_csv,
    write_json,
)
from stressid_pipeline import CORE_FEATURE_COLS, make_model  # noqa: E402


warnings.filterwarnings(
    "ignore",
    message=(
        r"X does not have valid feature names, but LGBMClassifier "
        r"was fitted with feature names"
    ),
)


MODELS = ["RandomForest", "XGBoost", "LightGBM"]
DESIGN_STANDARD = "standard_strict_LOSO"
DESIGN_DISJOINT = "subject_and_task_disjoint"
TARGET = "task_condition"
VIDEO_TASKS = {"Video1", "Video2"}
SCALAR_METRICS = [
    "balanced_accuracy",
    "macro_f1",
    "sensitivity",
    "specificity",
    "predicted_positive_rate",
]
REQUIRED_OUTPUTS = [
    "subject_task_disjoint_all_models.csv",
    "subject_task_disjoint_paired_differences.csv",
    "subject_task_disjoint_predictions.parquet",
    "subject_task_disjoint_task_diagnostics.csv",
    "subject_task_disjoint_fold_audit.csv",
    "subject_task_disjoint_summary.md",
    "subject_task_disjoint_numbers.json",
    "validation_checks.csv",
    "prior_outputs_manifest.csv",
    "prior_outputs_preserved.csv",
    "reproduction_commands.txt",
    "run.log",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_manifest(frame: pd.DataFrame, hash_col: str) -> str:
    digest = hashlib.sha256()
    for row in frame.sort_values("path").to_dict("records"):
        digest.update(f"{row['path']}|{row[hash_col]}\n".encode("utf-8"))
    return digest.hexdigest()


def output_guard(out_dir: Path) -> None:
    if out_dir.exists():
        existing = [path for path in out_dir.rglob("*") if path.is_file()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing files in "
                f"{out_dir}: {len(existing)} file(s)"
            )


def snapshot_prior_files(source_dir: Path, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    resolved_out = out_dir.resolve()
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved == resolved_out or resolved_out in resolved.parents:
            continue
        rows.append(
            {
                "path": str(path.relative_to(source_dir)).replace("\\", "/"),
                "size_before": int(path.stat().st_size),
                "sha256_before": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def verify_prior_files(
    source_dir: Path,
    before: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in before.to_dict("records"):
        path = source_dir / str(record["path"])
        exists = path.is_file()
        after_hash = sha256_file(path) if exists else ""
        rows.append(
            {
                **record,
                "exists_after": bool(exists),
                "size_after": int(path.stat().st_size) if exists else np.nan,
                "sha256_after": after_hash,
                "unchanged": bool(
                    exists and after_hash == record["sha256_before"]
                ),
            }
        )
    return pd.DataFrame(rows)


def binary_summary(
    truth: np.ndarray | pd.Series,
    pred: np.ndarray | pd.Series,
) -> dict[str, float | int | str]:
    y_true = np.asarray(truth, dtype=int)
    y_pred = np.asarray(pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    sensitivity = float(tp / (tp + fn)) if tp + fn else np.nan
    specificity = float(tn / (tn + fp)) if tn + fp else np.nan
    balanced = (
        float((sensitivity + specificity) / 2.0)
        if np.isfinite(sensitivity) and np.isfinite(specificity)
        else np.nan
    )
    return {
        "balanced_accuracy": balanced,
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "predicted_positive_rate": float(np.mean(y_pred == 1)),
        "confusion_matrix": json.dumps(cm.tolist(), separators=(",", ":")),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def metrics_from_confusions(
    matrices: np.ndarray,
) -> dict[str, np.ndarray]:
    cms = np.asarray(matrices, dtype=float)
    tn = cms[..., 0, 0]
    fp = cms[..., 0, 1]
    fn = cms[..., 1, 0]
    tp = cms[..., 1, 1]
    sensitivity = np.divide(
        tp,
        tp + fn,
        out=np.full_like(tp, np.nan, dtype=float),
        where=(tp + fn) > 0,
    )
    specificity = np.divide(
        tn,
        tn + fp,
        out=np.full_like(tn, np.nan, dtype=float),
        where=(tn + fp) > 0,
    )
    balanced = (sensitivity + specificity) / 2.0
    f1_0 = np.divide(
        2.0 * tn,
        2.0 * tn + fp + fn,
        out=np.zeros_like(tn, dtype=float),
        where=(2.0 * tn + fp + fn) > 0,
    )
    f1_1 = np.divide(
        2.0 * tp,
        2.0 * tp + fp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(2.0 * tp + fp + fn) > 0,
    )
    total = tn + fp + fn + tp
    predicted_positive_rate = np.divide(
        fp + tp,
        total,
        out=np.full_like(total, np.nan, dtype=float),
        where=total > 0,
    )
    return {
        "balanced_accuracy": balanced,
        "macro_f1": (f1_0 + f1_1) / 2.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "predicted_positive_rate": predicted_positive_rate,
    }


def subject_bootstrap_metric_cis(
    events: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> dict[str, tuple[float, float, int]]:
    subject_matrices = np.stack(
        [
            confusion_matrix(
                group["y_true"].to_numpy(dtype=int),
                group["y_pred"].to_numpy(dtype=int),
                labels=[0, 1],
            ).astype(float)
            for _, group in events.groupby("subject_id", sort=True)
        ],
        axis=0,
    )
    rng = np.random.default_rng(seed)
    n_subjects = subject_matrices.shape[0]
    sampled_subjects = rng.integers(
        0,
        n_subjects,
        size=(n_bootstrap, n_subjects),
    )
    selection_counts = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=n_subjects),
        1,
        sampled_subjects,
    )
    sampled_matrices = np.einsum(
        "bs,sij->bij",
        selection_counts,
        subject_matrices,
        optimize=True,
    )
    values = metrics_from_confusions(sampled_matrices)
    intervals: dict[str, tuple[float, float, int]] = {}
    for metric, vector in values.items():
        finite = vector[np.isfinite(vector)]
        intervals[metric] = (
            float(np.percentile(finite, 2.5)) if len(finite) else np.nan,
            float(np.percentile(finite, 97.5)) if len(finite) else np.nan,
            int(len(finite)),
        )
    return intervals


def paired_subject_bootstrap(
    paired: pd.DataFrame,
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float, int]:
    groups = list(paired.groupby("subject_id", sort=True))
    standard_matrices = np.stack(
        [
            confusion_matrix(
                group["y_true"].to_numpy(dtype=int),
                group["y_pred_standard"].to_numpy(dtype=int),
                labels=[0, 1],
            ).astype(float)
            for _, group in groups
        ],
        axis=0,
    )
    disjoint_matrices = np.stack(
        [
            confusion_matrix(
                group["y_true"].to_numpy(dtype=int),
                group["y_pred_disjoint"].to_numpy(dtype=int),
                labels=[0, 1],
            ).astype(float)
            for _, group in groups
        ],
        axis=0,
    )
    point_standard = binary_summary(
        paired["y_true"],
        paired["y_pred_standard"],
    )[metric]
    point_disjoint = binary_summary(
        paired["y_true"],
        paired["y_pred_disjoint"],
    )[metric]
    point = float(point_disjoint) - float(point_standard)
    rng = np.random.default_rng(seed)
    n_subjects = len(groups)
    sampled_subjects = rng.integers(
        0,
        n_subjects,
        size=(n_bootstrap, n_subjects),
    )
    counts = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=n_subjects),
        1,
        sampled_subjects,
    )
    sampled_standard = np.einsum(
        "bs,sij->bij",
        counts,
        standard_matrices,
        optimize=True,
    )
    sampled_disjoint = np.einsum(
        "bs,sij->bij",
        counts,
        disjoint_matrices,
        optimize=True,
    )
    standard_values = metrics_from_confusions(sampled_standard)[metric]
    disjoint_values = metrics_from_confusions(sampled_disjoint)[metric]
    differences = disjoint_values - standard_values
    finite = differences[np.isfinite(differences)]
    return (
        point,
        float(np.percentile(finite, 2.5)) if len(finite) else np.nan,
        float(np.percentile(finite, 97.5)) if len(finite) else np.nan,
        int(len(finite)),
    )


def prepare_windows(features_path: Path) -> pd.DataFrame:
    features = pd.read_csv(features_path)
    missing = [
        column
        for column in [
            "window_id",
            "subject_id",
            "task_id",
            "start_sec",
            "end_sec",
            TARGET,
            *CORE_FEATURE_COLS,
        ]
        if column not in features.columns
    ]
    if missing:
        raise RuntimeError(f"missing required feature columns: {missing}")
    windows = features[
        features[TARGET].notna()
        & ~features["task_id"].astype(str).isin(VIDEO_TASKS)
    ].copy()
    windows["subject_id"] = windows["subject_id"].astype(str)
    windows["task_id"] = windows["task_id"].astype(str)
    windows["window_id"] = windows["window_id"].astype(str)
    windows["event_id"] = (
        windows["subject_id"] + "|" + windows["task_id"]
    )
    windows["y_true"] = windows[TARGET].astype(int)
    # Preserve source-row order because it is part of the retained fixed-model
    # computation and affects exact Random Forest probability reproduction.
    windows = windows.reset_index(drop=True)
    if windows[["subject_id", "task_id", "window_id"]].duplicated().any():
        raise RuntimeError("feature window keys are not unique")
    event_truth_counts = windows.groupby("event_id")["y_true"].nunique()
    if not bool((event_truth_counts == 1).all()):
        raise RuntimeError("task-condition truth varies within an event")
    if set(windows["y_true"].unique()) != {0, 1}:
        raise RuntimeError("eligible windows do not contain both classes")
    return windows


def training_metadata(
    windows: pd.DataFrame,
    train_mask: np.ndarray,
) -> dict[str, object]:
    train = windows.loc[train_mask]
    event_labels = train[
        ["subject_id", "task_id", "event_id", "y_true"]
    ].drop_duplicates()
    subjects = sorted(train["subject_id"].unique())
    tasks = sorted(train["task_id"].unique())
    return {
        "training_subjects": ";".join(subjects),
        "n_training_subjects": int(len(subjects)),
        "training_tasks": ";".join(tasks),
        "n_training_tasks": int(len(tasks)),
        "train_class_0_windows": int((train["y_true"] == 0).sum()),
        "train_class_1_windows": int((train["y_true"] == 1).sum()),
        "train_class_0_events": int(
            (event_labels["y_true"] == 0).sum()
        ),
        "train_class_1_events": int(
            (event_labels["y_true"] == 1).sum()
        ),
        "both_training_classes_present": bool(
            train["y_true"].nunique() == 2
        ),
    }


def prediction_columns() -> list[str]:
    return [
        "subject_id",
        "task_id",
        "event_id",
        "window_id",
        "start_sec",
        "end_sec",
        "target",
        "model",
        "evaluation_design",
        "fold_id",
        "prediction_level",
        "aggregation",
        "y_true",
        "y_score",
        "y_pred",
        "n_windows",
        "prediction_source",
    ]


def make_standard_predictions(
    windows: pd.DataFrame,
    retained_path: Path,
    models: list[str],
    logger: RunLogger,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    retained = pd.read_parquet(retained_path)
    retained["subject_id"] = retained["subject_id"].astype(str)
    retained["task_id"] = retained["task_id"].astype(str)
    retained["window_id"] = retained["window_id"].astype(str)
    retained["event_id"] = (
        retained["subject_id"] + "|" + retained["task_id"]
    )
    expected_keys = set(
        zip(
            windows["subject_id"],
            windows["task_id"],
            windows["window_id"],
        )
    )
    prediction_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []

    for model_name in models:
        selected = retained[
            (retained["target"] == TARGET)
            & (retained["model"] == model_name)
            & (retained["prediction_level"] == "window")
        ].copy()
        selected_keys = set(
            zip(
                selected["subject_id"],
                selected["task_id"],
                selected["window_id"],
            )
        )
        keys_match = selected_keys == expected_keys
        checks.append(
            {
                "check_id": f"standard_window_keys_{model_name}",
                "description": (
                    "retained standard-LOSO window keys match eligible "
                    "feature windows"
                ),
                "passed": bool(keys_match),
                "detail": (
                    f"expected={len(expected_keys)}; "
                    f"observed={len(selected_keys)}"
                ),
            }
        )
        if not keys_match:
            raise RuntimeError(
                f"retained standard keys mismatch for {model_name}"
            )
        selected = selected.merge(
            windows[
                [
                    "subject_id",
                    "task_id",
                    "window_id",
                    "event_id",
                    "start_sec",
                    "end_sec",
                    "y_true",
                ]
            ],
            on=["subject_id", "task_id", "window_id"],
            how="inner",
            suffixes=("_retained", ""),
            validate="one_to_one",
        )
        truth_match = np.array_equal(
            selected["y_true_retained"].to_numpy(dtype=int),
            selected["y_true"].to_numpy(dtype=int),
        )
        checks.append(
            {
                "check_id": f"standard_truth_{model_name}",
                "description": (
                    "retained standard-LOSO truth matches feature truth"
                ),
                "passed": bool(truth_match),
                "detail": f"n_windows={len(selected)}",
            }
        )
        if not truth_match:
            raise RuntimeError(
                f"retained standard truth mismatch for {model_name}"
            )
        standard_windows = pd.DataFrame(
            {
                "subject_id": selected["subject_id"],
                "task_id": selected["task_id"],
                "event_id": selected["event_id"],
                "window_id": selected["window_id"],
                "start_sec": selected["start_sec"],
                "end_sec": selected["end_sec"],
                "target": TARGET,
                "model": model_name,
                "evaluation_design": DESIGN_STANDARD,
                "fold_id": selected["subject_id"],
                "prediction_level": "window",
                "aggregation": "",
                "y_true": selected["y_true"],
                "y_score": selected["y_score"].astype(float),
                "y_pred": (
                    selected["y_score"].astype(float) >= 0.5
                ).astype(int),
                "n_windows": 1,
                "prediction_source": (
                    "retained_stressid_window_oof"
                ),
            }
        )
        standard_events = (
            standard_windows.groupby(
                [
                    "subject_id",
                    "task_id",
                    "event_id",
                    "target",
                    "model",
                    "evaluation_design",
                    "fold_id",
                ],
                as_index=False,
                sort=True,
            )
            .agg(
                y_true=("y_true", "first"),
                y_score=("y_score", "mean"),
                n_windows=("window_id", "size"),
            )
        )
        standard_events["window_id"] = ""
        standard_events["start_sec"] = np.nan
        standard_events["end_sec"] = np.nan
        standard_events["prediction_level"] = "participant_task"
        standard_events["aggregation"] = "arithmetic_mean"
        standard_events["y_pred"] = (
            standard_events["y_score"] >= 0.5
        ).astype(int)
        standard_events["prediction_source"] = (
            "reaggregated_retained_stressid_window_oof"
        )

        retained_events = retained[
            (retained["target"] == TARGET)
            & (retained["model"] == model_name)
            & (retained["prediction_level"] == "event")
            & (retained["aggregation"] == "mean_probability")
        ][["event_id", "y_score", "y_pred", "n_windows"]].copy()
        comparison = standard_events[
            ["event_id", "y_score", "y_pred", "n_windows"]
        ].merge(
            retained_events,
            on="event_id",
            suffixes=("_new", "_retained"),
            validate="one_to_one",
        )
        max_score_error = float(
            np.max(
                np.abs(
                    comparison["y_score_new"]
                    - comparison["y_score_retained"]
                )
            )
        )
        retained_event_match = bool(
            len(comparison) == len(standard_events)
            and max_score_error <= 1e-12
            and (
                comparison["y_pred_new"]
                == comparison["y_pred_retained"]
            ).all()
            and (
                comparison["n_windows_new"]
                == comparison["n_windows_retained"]
            ).all()
        )
        checks.append(
            {
                "check_id": f"standard_event_reproduction_{model_name}",
                "description": (
                    "arithmetic mean of retained windows reproduces "
                    "retained standard-LOSO event predictions"
                ),
                "passed": retained_event_match,
                "detail": (
                    f"n_events={len(comparison)}; "
                    f"max_abs_probability_error={max_score_error:.3e}"
                ),
            }
        )
        if not retained_event_match:
            raise RuntimeError(
                f"standard event reproduction failed for {model_name}"
            )

        prediction_parts.extend([standard_windows, standard_events])
        for event in standard_events.to_dict("records"):
            test_subject = str(event["subject_id"])
            test_task = str(event["task_id"])
            train_mask = (
                windows["subject_id"].to_numpy() != test_subject
            )
            metadata = training_metadata(windows, train_mask)
            event_windows = standard_windows[
                standard_windows["event_id"] == event["event_id"]
            ]
            audit_rows.append(
                {
                    "model": model_name,
                    "evaluation_design": DESIGN_STANDARD,
                    "fold_id": test_subject,
                    "test_subject_id": test_subject,
                    "test_task_id": test_task,
                    "event_id": event["event_id"],
                    "y_true": int(event["y_true"]),
                    "n_test_windows": int(event["n_windows"]),
                    **metadata,
                    "test_subject_absent_from_training": bool(
                        not (
                            windows.loc[
                                train_mask,
                                "subject_id",
                            ]
                            == test_subject
                        ).any()
                    ),
                    "exact_test_task_absent_from_training": bool(
                        not (
                            windows.loc[train_mask, "task_id"]
                            == test_task
                        ).any()
                    ),
                    "imputation_fit_scope": "outer_training_rows_only",
                    "scaling_fit_scope": "outer_training_rows_only",
                    "feature_selection": "none",
                    "feature_selection_fit_scope": "not_applicable",
                    "class_weight": "none",
                    "class_handling": (
                        "sample_weight=1/windows_in_subject_task"
                    ),
                    "model_fit_scope": "outer_training_subjects_only",
                    "fold_status": "loaded_retained_oof",
                    "failure_reason": "",
                    "window_prediction_unique_classes": int(
                        event_windows["y_pred"].nunique()
                    ),
                    "window_prediction_single_class": bool(
                        event_windows["y_pred"].nunique() == 1
                    ),
                    "window_score_min": float(
                        event_windows["y_score"].min()
                    ),
                    "window_score_max": float(
                        event_windows["y_score"].max()
                    ),
                    "event_score": float(event["y_score"]),
                    "event_mean_window_score": float(
                        event_windows["y_score"].mean()
                    ),
                    "event_mean_abs_error": float(
                        abs(
                            event["y_score"]
                            - event_windows["y_score"].mean()
                        )
                    ),
                    "probabilities_in_range": bool(
                        event_windows["y_score"].between(0, 1).all()
                    ),
                }
            )
        logger(
            f"loaded and reaggregated standard LOSO {model_name}: "
            f"{len(standard_windows)} windows, "
            f"{len(standard_events)} events"
        )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    return (
        predictions[prediction_columns()],
        pd.DataFrame(audit_rows),
        checks,
    )


def make_disjoint_predictions(
    windows: pd.DataFrame,
    models: list[str],
    logger: RunLogger,
    event_limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    event_keys = (
        windows[["subject_id", "task_id", "event_id", "y_true"]]
        .drop_duplicates()
        .sort_values(["subject_id", "task_id"])
        .reset_index(drop=True)
    )
    if event_limit is not None:
        event_keys = event_keys.iloc[:event_limit].copy()
    subjects = windows["subject_id"].to_numpy(dtype=str)
    tasks = windows["task_id"].to_numpy(dtype=str)
    raw = windows[CORE_FEATURE_COLS].to_numpy(dtype=float)
    prediction_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    start_all = time.perf_counter()

    for event_number, event in enumerate(
        event_keys.to_dict("records"),
        start=1,
    ):
        test_subject = str(event["subject_id"])
        test_task = str(event["task_id"])
        event_id = str(event["event_id"])
        train_mask = (subjects != test_subject) & (tasks != test_task)
        test_mask = (subjects == test_subject) & (tasks == test_task)
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)
        metadata = training_metadata(windows, train_mask)
        subject_absent = bool(
            test_subject not in set(subjects[train_idx])
        )
        task_absent = bool(test_task not in set(tasks[train_idx]))
        base_audit = {
            "evaluation_design": DESIGN_DISJOINT,
            "fold_id": event_id,
            "test_subject_id": test_subject,
            "test_task_id": test_task,
            "event_id": event_id,
            "y_true": int(event["y_true"]),
            "n_test_windows": int(len(test_idx)),
            **metadata,
            "test_subject_absent_from_training": subject_absent,
            "exact_test_task_absent_from_training": task_absent,
            "imputation_fit_scope": "training_rows_only",
            "scaling_fit_scope": "training_rows_only",
            "feature_selection": "none",
            "feature_selection_fit_scope": "not_applicable",
            "class_weight": "none",
            "class_handling": (
                "sample_weight=1/windows_in_subject_task"
            ),
            "model_fit_scope": (
                "records_excluding_test_subject_and_exact_test_task"
            ),
        }
        if not len(test_idx):
            for model_name in models:
                audit_rows.append(
                    {
                        **base_audit,
                        "model": model_name,
                        "fold_status": "failed_no_test_windows",
                        "failure_reason": "no held-out event windows",
                    }
                )
            continue
        if not metadata["both_training_classes_present"]:
            for model_name in models:
                audit_rows.append(
                    {
                        **base_audit,
                        "model": model_name,
                        "fold_status": "failed_single_training_class",
                        "failure_reason": (
                            "training data lacks one binary target class"
                        ),
                    }
                )
            continue
        if not subject_absent or not task_absent:
            raise RuntimeError(
                f"exclusion guard failed for {event_id}: "
                f"subject_absent={subject_absent}; task_absent={task_absent}"
            )

        try:
            x_train, x_test, medians, means, scales = (
                robust_impute_scale(raw[train_idx], raw[test_idx])
            )
            weights = event_equal_weights(
                windows.iloc[train_idx],
                ["subject_id", "task_id"],
            )
            y_train = windows.iloc[train_idx]["y_true"].to_numpy(
                dtype=int
            )
            if not (
                np.isfinite(x_train).all()
                and np.isfinite(x_test).all()
                and np.isfinite(weights).all()
            ):
                raise RuntimeError("non-finite preprocessed values")
            preprocessing_detail = {
                "imputer_median_sha256": hashlib.sha256(
                    medians.tobytes()
                ).hexdigest(),
                "scaler_mean_sha256": hashlib.sha256(
                    means.tobytes()
                ).hexdigest(),
                "scaler_scale_sha256": hashlib.sha256(
                    scales.tobytes()
                ).hexdigest(),
                "sample_weight_min": float(weights.min()),
                "sample_weight_max": float(weights.max()),
            }
        except Exception as exc:
            logger.exception(f"preprocessing {event_id}", exc)
            for model_name in models:
                audit_rows.append(
                    {
                        **base_audit,
                        "model": model_name,
                        "fold_status": "failed_preprocessing",
                        "failure_reason": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                )
            continue

        for model_name in models:
            try:
                classifier = make_model(model_name)
                if classifier is None:
                    raise RuntimeError(f"{model_name} unavailable")
                classifier.fit(
                    x_train,
                    y_train,
                    sample_weight=weights,
                )
                class_index = list(classifier.classes_).index(1)
                scores = np.asarray(
                    classifier.predict_proba(x_test)[:, class_index],
                    dtype=float,
                )
                if not np.isfinite(scores).all():
                    raise RuntimeError("non-finite predicted probability")
                if not bool(
                    np.logical_and(scores >= 0, scores <= 1).all()
                ):
                    raise RuntimeError("probability outside [0, 1]")
                event_windows = windows.iloc[test_idx][
                    [
                        "subject_id",
                        "task_id",
                        "event_id",
                        "window_id",
                        "start_sec",
                        "end_sec",
                        "y_true",
                    ]
                ].copy()
                event_windows["target"] = TARGET
                event_windows["model"] = model_name
                event_windows["evaluation_design"] = DESIGN_DISJOINT
                event_windows["fold_id"] = event_id
                event_windows["prediction_level"] = "window"
                event_windows["aggregation"] = ""
                event_windows["y_score"] = scores
                event_windows["y_pred"] = (scores >= 0.5).astype(int)
                event_windows["n_windows"] = 1
                event_windows["prediction_source"] = (
                    "new_subject_and_task_disjoint_fit"
                )
                event_score = float(scores.mean())
                event_prediction = pd.DataFrame(
                    [
                        {
                            "subject_id": test_subject,
                            "task_id": test_task,
                            "event_id": event_id,
                            "window_id": "",
                            "start_sec": np.nan,
                            "end_sec": np.nan,
                            "target": TARGET,
                            "model": model_name,
                            "evaluation_design": DESIGN_DISJOINT,
                            "fold_id": event_id,
                            "prediction_level": "participant_task",
                            "aggregation": "arithmetic_mean",
                            "y_true": int(event["y_true"]),
                            "y_score": event_score,
                            "y_pred": int(event_score >= 0.5),
                            "n_windows": int(len(test_idx)),
                            "prediction_source": (
                                "mean_of_disjoint_window_probabilities"
                            ),
                        }
                    ]
                )
                prediction_parts.extend(
                    [
                        event_windows[prediction_columns()],
                        event_prediction[prediction_columns()],
                    ]
                )
                audit_rows.append(
                    {
                        **base_audit,
                        **preprocessing_detail,
                        "model": model_name,
                        "fold_status": "completed",
                        "failure_reason": "",
                        "window_prediction_unique_classes": int(
                            event_windows["y_pred"].nunique()
                        ),
                        "window_prediction_single_class": bool(
                            event_windows["y_pred"].nunique() == 1
                        ),
                        "window_score_min": float(scores.min()),
                        "window_score_max": float(scores.max()),
                        "event_score": event_score,
                        "event_mean_window_score": float(scores.mean()),
                        "event_mean_abs_error": float(
                            abs(event_score - scores.mean())
                        ),
                        "probabilities_in_range": bool(
                            np.logical_and(
                                scores >= 0,
                                scores <= 1,
                            ).all()
                        ),
                    }
                )
            except Exception as exc:
                logger.exception(
                    f"{model_name} disjoint {event_id}",
                    exc,
                )
                audit_rows.append(
                    {
                        **base_audit,
                        **preprocessing_detail,
                        "model": model_name,
                        "fold_status": "failed_model_fit_or_prediction",
                        "failure_reason": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                )

        if (
            event_number == 1
            or event_number % 10 == 0
            or event_number == len(event_keys)
        ):
            elapsed = time.perf_counter() - start_all
            logger(
                "subject-task-disjoint events "
                f"{event_number}/{len(event_keys)}; "
                f"elapsed={elapsed:.1f}s"
            )

    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        if prediction_parts
        else pd.DataFrame(columns=prediction_columns())
    )
    audit = pd.DataFrame(audit_rows)
    checks.extend(
        [
            {
                "check_id": "disjoint_test_subject_absent",
                "description": (
                    "test participant is absent from every completed "
                    "disjoint training fold"
                ),
                "passed": bool(
                    audit.loc[
                        audit["fold_status"] == "completed",
                        "test_subject_absent_from_training",
                    ].all()
                ),
                "detail": (
                    f"completed_audit_rows="
                    f"{int((audit['fold_status'] == 'completed').sum())}"
                ),
            },
            {
                "check_id": "disjoint_exact_task_absent",
                "description": (
                    "exact test task is absent from every completed "
                    "disjoint training fold"
                ),
                "passed": bool(
                    audit.loc[
                        audit["fold_status"] == "completed",
                        "exact_test_task_absent_from_training",
                    ].all()
                ),
                "detail": (
                    f"completed_audit_rows="
                    f"{int((audit['fold_status'] == 'completed').sum())}"
                ),
            },
            {
                "check_id": "disjoint_training_classes",
                "description": (
                    "both target classes remain in completed training folds"
                ),
                "passed": bool(
                    audit.loc[
                        audit["fold_status"] == "completed",
                        "both_training_classes_present",
                    ].all()
                ),
                "detail": (
                    "single_class_training_rows="
                    f"{int((~audit['both_training_classes_present']).sum())}"
                ),
            },
        ]
    )
    return predictions, audit, checks


def summarize_performance(
    event_predictions: pd.DataFrame,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_name, design), group in event_predictions.groupby(
        ["model", "evaluation_design"],
        sort=True,
    ):
        summary = binary_summary(group["y_true"], group["y_pred"])
        intervals = subject_bootstrap_metric_cis(
            group,
            n_bootstrap,
            stable_seed(
                "subject_task_disjoint_all_models",
                model_name,
                design,
            ),
        )
        common = {
            "dataset": "StressID",
            "target": TARGET,
            "model": model_name,
            "evaluation_design": design,
            "evaluation_unit": "participant_task",
            "aggregation": "arithmetic_mean_window_probability",
            "bootstrap_unit": "participant",
            "n_bootstrap": n_bootstrap,
            "n_participants": int(group["subject_id"].nunique()),
            "n_participant_task_events": int(len(group)),
            "n_unique_heldout_tasks": int(group["task_id"].nunique()),
            "n_windows": int(group["n_windows"].sum()),
            "tn": int(summary["tn"]),
            "fp": int(summary["fp"]),
            "fn": int(summary["fn"]),
            "tp": int(summary["tp"]),
            "confusion_matrix": summary["confusion_matrix"],
        }
        for metric in SCALAR_METRICS:
            low, high, valid = intervals[metric]
            rows.append(
                {
                    **common,
                    "metric": metric,
                    "estimate": float(summary[metric]),
                    "estimate_text": "",
                    "ci_low": low,
                    "ci_high": high,
                    "valid_bootstrap_resamples": valid,
                }
            )
        rows.append(
            {
                **common,
                "metric": "confusion_matrix",
                "estimate": np.nan,
                "estimate_text": summary["confusion_matrix"],
                "ci_low": np.nan,
                "ci_high": np.nan,
                "valid_bootstrap_resamples": 0,
            }
        )
    return pd.DataFrame(rows)


def paired_differences(
    event_predictions: pd.DataFrame,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in MODELS:
        selected = event_predictions[
            event_predictions["model"] == model_name
        ]
        standard = selected[
            selected["evaluation_design"] == DESIGN_STANDARD
        ][
            [
                "subject_id",
                "task_id",
                "event_id",
                "y_true",
                "y_pred",
                "y_score",
            ]
        ].rename(
            columns={
                "y_pred": "y_pred_standard",
                "y_score": "y_score_standard",
            }
        )
        disjoint = selected[
            selected["evaluation_design"] == DESIGN_DISJOINT
        ][
            [
                "subject_id",
                "task_id",
                "event_id",
                "y_true",
                "y_pred",
                "y_score",
            ]
        ].rename(
            columns={
                "y_true": "y_true_disjoint",
                "y_pred": "y_pred_disjoint",
                "y_score": "y_score_disjoint",
            }
        )
        paired = standard.merge(
            disjoint,
            on=["subject_id", "task_id", "event_id"],
            how="inner",
            validate="one_to_one",
        )
        if not np.array_equal(
            paired["y_true"].to_numpy(dtype=int),
            paired["y_true_disjoint"].to_numpy(dtype=int),
        ):
            raise RuntimeError(
                f"paired truth mismatch for {model_name}"
            )
        for metric in ["balanced_accuracy", "macro_f1"]:
            point, low, high, valid = paired_subject_bootstrap(
                paired,
                metric,
                n_bootstrap,
                stable_seed(
                    "subject_task_disjoint_paired",
                    model_name,
                    metric,
                ),
            )
            standard_value = float(
                binary_summary(
                    paired["y_true"],
                    paired["y_pred_standard"],
                )[metric]
            )
            disjoint_value = float(
                binary_summary(
                    paired["y_true"],
                    paired["y_pred_disjoint"],
                )[metric]
            )
            rows.append(
                {
                    "dataset": "StressID",
                    "target": TARGET,
                    "model": model_name,
                    "metric": metric,
                    "comparison": (
                        "subject_and_task_disjoint_minus_"
                        "standard_strict_LOSO"
                    ),
                    "standard_estimate_on_common_events": standard_value,
                    "disjoint_estimate_on_common_events": disjoint_value,
                    "point_difference": point,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_unit": "participant",
                    "n_bootstrap": n_bootstrap,
                    "valid_bootstrap_resamples": valid,
                    "n_paired_participants": int(
                        paired["subject_id"].nunique()
                    ),
                    "n_paired_participant_task_events": int(
                        len(paired)
                    ),
                    "paired_keys_exact": bool(
                        len(paired) == len(standard) == len(disjoint)
                    ),
                }
            )
    return pd.DataFrame(rows)


def task_diagnostics(
    event_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_name, design, task_name), group in (
        event_predictions.groupby(
            ["model", "evaluation_design", "task_id"],
            sort=True,
        )
    ):
        summary = binary_summary(group["y_true"], group["y_pred"])
        both_targets = group["y_true"].nunique() == 2
        rows.append(
            {
                "dataset": "StressID",
                "target": TARGET,
                "model": model_name,
                "evaluation_design": design,
                "task_name": task_name,
                "binary_class": (
                    int(group["y_true"].iloc[0])
                    if group["y_true"].nunique() == 1
                    else "mixed"
                ),
                "n_participants": int(group["subject_id"].nunique()),
                "n_events": int(len(group)),
                "balanced_accuracy": (
                    float(summary["balanced_accuracy"])
                    if both_targets
                    else np.nan
                ),
                "balanced_accuracy_defined": bool(both_targets),
                "balanced_accuracy_undefined_reason": (
                    ""
                    if both_targets
                    else "only_one_target_class_present_for_exact_task"
                ),
                "macro_f1": float(summary["macro_f1"]),
                "sensitivity": float(summary["sensitivity"]),
                "specificity": float(summary["specificity"]),
                "predicted_positive_rate": float(
                    summary["predicted_positive_rate"]
                ),
                "confusion_matrix": summary["confusion_matrix"],
                "tn": int(summary["tn"]),
                "fp": int(summary["fp"]),
                "fn": int(summary["fn"]),
                "tp": int(summary["tp"]),
                "descriptive_only": True,
                "small_sample_flag": bool(len(group) < 30),
            }
        )
    return pd.DataFrame(rows)


def build_validation_checks(
    predictions: pd.DataFrame,
    audit: pd.DataFrame,
    standard_checks: list[dict[str, object]],
    disjoint_checks: list[dict[str, object]],
    prior_rf_path: Path,
    preserved: pd.DataFrame,
) -> pd.DataFrame:
    checks = [*standard_checks, *disjoint_checks]
    event_predictions = predictions[
        predictions["prediction_level"] == "participant_task"
    ].copy()
    window_predictions = predictions[
        predictions["prediction_level"] == "window"
    ].copy()
    event_means = (
        window_predictions.groupby(
            [
                "subject_id",
                "task_id",
                "event_id",
                "model",
                "evaluation_design",
            ],
            as_index=False,
        )["y_score"]
        .mean()
        .rename(columns={"y_score": "window_mean"})
    )
    mean_check = event_predictions.merge(
        event_means,
        on=[
            "subject_id",
            "task_id",
            "event_id",
            "model",
            "evaluation_design",
        ],
        how="left",
        validate="one_to_one",
    )
    max_mean_error = float(
        np.max(np.abs(mean_check["y_score"] - mean_check["window_mean"]))
    )
    checks.append(
        {
            "check_id": "event_arithmetic_mean",
            "description": (
                "every event probability equals the arithmetic mean of "
                "its constituent window probabilities"
            ),
            "passed": bool(max_mean_error <= 1e-12),
            "detail": f"max_abs_error={max_mean_error:.3e}",
        }
    )
    key_columns = [
        "model",
        "evaluation_design",
        "prediction_level",
        "subject_id",
        "task_id",
        "window_id",
    ]
    duplicate_count = int(predictions.duplicated(key_columns).sum())
    checks.append(
        {
            "check_id": "prediction_keys_unique",
            "description": "prediction keys are unique",
            "passed": bool(duplicate_count == 0),
            "detail": f"duplicate_rows={duplicate_count}",
        }
    )
    in_range = bool(predictions["y_score"].between(0, 1).all())
    checks.append(
        {
            "check_id": "probabilities_in_range",
            "description": "all predicted probabilities are in [0, 1]",
            "passed": in_range,
            "detail": (
                f"min={predictions['y_score'].min():.12f}; "
                f"max={predictions['y_score'].max():.12f}"
            ),
        }
    )
    completed_audit = audit[audit["fold_status"].isin(
        ["completed", "loaded_retained_oof"]
    )]
    preprocessing_scope_ok = bool(
        completed_audit["imputation_fit_scope"]
        .astype(str)
        .str.contains("training")
        .all()
        and completed_audit["scaling_fit_scope"]
        .astype(str)
        .str.contains("training")
        .all()
        and (completed_audit["feature_selection"] == "none").all()
        and (completed_audit["class_weight"] == "none").all()
    )
    checks.append(
        {
            "check_id": "training_only_preprocessing",
            "description": (
                "imputation, scaling, feature handling, and class handling "
                "are recorded as training-only"
            ),
            "passed": preprocessing_scope_ok,
            "detail": (
                "feature_selection=none; class_weight=none; "
                "event-equal sample weights use training rows"
            ),
        }
    )

    prior_rf = pd.read_parquet(prior_rf_path).copy()
    prior_rf["subject_id"] = prior_rf["subject_id"].astype(str)
    prior_rf["task_id"] = prior_rf["task_id"].astype(str)
    prior_rf["event_id"] = (
        prior_rf["subject_id"] + "|" + prior_rf["task_id"]
    )
    new_rf = event_predictions[
        (event_predictions["model"] == "RandomForest")
        & (
            event_predictions["evaluation_design"]
            == DESIGN_DISJOINT
        )
    ][["event_id", "y_score", "y_pred"]]
    rf_compare = new_rf.merge(
        prior_rf[["event_id", "y_score", "y_pred"]],
        on="event_id",
        suffixes=("_new", "_prior"),
        validate="one_to_one",
    )
    rf_max_error = float(
        np.max(
            np.abs(
                rf_compare["y_score_new"] - rf_compare["y_score_prior"]
            )
        )
    )
    rf_reproduced = bool(
        len(rf_compare) == len(prior_rf)
        and rf_max_error <= 1e-12
        and (
            rf_compare["y_pred_new"] == rf_compare["y_pred_prior"]
        ).all()
    )
    checks.append(
        {
            "check_id": "prior_random_forest_reproduction",
            "description": (
                "Random Forest subject-and-task-disjoint event "
                "probabilities reproduce the prior addendum"
            ),
            "passed": rf_reproduced,
            "detail": (
                f"n_events={len(rf_compare)}; "
                f"max_abs_probability_error={rf_max_error:.3e}"
            ),
        }
    )
    unchanged_count = int(preserved["unchanged"].sum())
    checks.append(
        {
            "check_id": "prior_outputs_hash_identical",
            "description": (
                "all files existing under paper_results_v2 before this "
                "run remain SHA-256 identical"
            ),
            "passed": bool(unchanged_count == len(preserved)),
            "detail": (
                f"unchanged={unchanged_count}; "
                f"checked={len(preserved)}"
            ),
        }
    )
    return pd.DataFrame(checks)


def make_numbers(
    performance: pd.DataFrame,
    paired: pd.DataFrame,
) -> dict[str, object]:
    numbers: dict[str, object] = {
        "analysis_random_seed": RANDOM_STATE,
        "analysis_bootstrap_resamples": int(
            performance["n_bootstrap"].max()
        ),
    }
    model_slugs = {
        "RandomForest": "rf",
        "XGBoost": "xgb",
        "LightGBM": "lgbm",
    }
    design_slugs = {
        DESIGN_STANDARD: "standard_loso",
        DESIGN_DISJOINT: "subject_task_disjoint",
    }
    for row in performance[
        performance["metric"].isin(
            ["balanced_accuracy", "macro_f1"]
        )
    ].to_dict("records"):
        prefix = (
            f"stressid_task_{design_slugs[row['evaluation_design']]}_"
            f"{model_slugs[row['model']]}_{row['metric']}"
        )
        numbers[prefix] = row["estimate"]
        numbers[f"{prefix}_ci_low"] = row["ci_low"]
        numbers[f"{prefix}_ci_high"] = row["ci_high"]
        numbers[f"{prefix}_n_participants"] = row["n_participants"]
        numbers[f"{prefix}_n_events"] = row[
            "n_participant_task_events"
        ]
    for row in paired.to_dict("records"):
        prefix = (
            "stressid_task_subject_task_disjoint_minus_standard_loso_"
            f"{model_slugs[row['model']]}_{row['metric']}"
        )
        numbers[prefix] = row["point_difference"]
        numbers[f"{prefix}_ci_low"] = row["ci_low"]
        numbers[f"{prefix}_ci_high"] = row["ci_high"]
    return numbers


def format_ci(row: pd.Series) -> str:
    return (
        f"{float(row['estimate']):.3f} "
        f"[{float(row['ci_low']):.3f}, "
        f"{float(row['ci_high']):.3f}]"
    )


def write_summary(
    path: Path,
    performance: pd.DataFrame,
    paired: pd.DataFrame,
    task_table: pd.DataFrame,
    audit: pd.DataFrame,
    checks: pd.DataFrame,
    manifest_before_digest: str,
    manifest_after_digest: str,
) -> None:
    primary = performance[
        performance["metric"].isin(
            ["balanced_accuracy", "macro_f1"]
        )
    ].copy()
    pivot_rows: list[str] = []
    for model_name in MODELS:
        fields: list[str] = []
        for design in [DESIGN_STANDARD, DESIGN_DISJOINT]:
            for metric in ["balanced_accuracy", "macro_f1"]:
                row = primary[
                    (primary["model"] == model_name)
                    & (primary["evaluation_design"] == design)
                    & (primary["metric"] == metric)
                ].iloc[0]
                fields.append(format_ci(row))
        pivot_rows.append(
            "| "
            + " | ".join([model_name, *fields])
            + " |"
        )
    paired_rows: list[str] = []
    for row in paired.to_dict("records"):
        paired_rows.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["metric"]),
                    f"{float(row['point_difference']):.3f}",
                    (
                        f"[{float(row['ci_low']):.3f}, "
                        f"{float(row['ci_high']):.3f}]"
                    ),
                    str(row["n_paired_participants"]),
                    str(row["n_paired_participant_task_events"]),
                ]
            )
            + " |"
        )
    disjoint_ba = primary[
        (primary["evaluation_design"] == DESIGN_DISJOINT)
        & (primary["metric"] == "balanced_accuracy")
    ].set_index("model")
    standard_ba = primary[
        (primary["evaluation_design"] == DESIGN_STANDARD)
        & (primary["metric"] == "balanced_accuracy")
    ].set_index("model")
    declines = {
        model: bool(
            disjoint_ba.loc[model, "estimate"]
            < standard_ba.loc[model, "estimate"]
        )
        for model in MODELS
    }
    toward_half = {
        model: bool(
            abs(disjoint_ba.loc[model, "estimate"] - 0.5)
            < abs(standard_ba.loc[model, "estimate"] - 0.5)
        )
        for model in MODELS
    }
    if all(declines.values()) and all(toward_half.values()):
        consistency_sentence = (
            "Across all three prespecified tree models, the "
            "subject-and-task-disjoint estimate declined toward 0.5. "
            "Standard LOSO task-condition performance depended strongly "
            "on task types represented during training."
        )
    else:
        consistency_sentence = (
            "The change under unseen-task transfer was model-dependent."
        )
    exact_estimates = "; ".join(
        (
            f"{model} {disjoint_ba.loc[model, 'estimate']:.3f} "
            f"[{disjoint_ba.loc[model, 'ci_low']:.3f}, "
            f"{disjoint_ba.loc[model, 'ci_high']:.3f}]"
        )
        for model in MODELS
    )
    failed = audit[
        ~audit["fold_status"].isin(
            ["completed", "loaded_retained_oof"]
        )
    ]
    single_class_train = audit[
        ~audit["both_training_classes_present"].fillna(False)
    ]
    disjoint_collapsed = audit[
        (audit["evaluation_design"] == DESIGN_DISJOINT)
        & (audit["fold_status"] == "completed")
        & audit["window_prediction_single_class"].fillna(False)
    ]
    one_class_tasks = task_table[
        ~task_table["balanced_accuracy_defined"]
    ]["task_name"].drop_duplicates()
    lines = [
        "# StressID Subject-and-Task-Disjoint Tree-Model Addendum",
        "",
        "## Design",
        "",
        (
            "Dataset: StressID. Target: protocol-defined binary task "
            "condition. Unit: participant-task. Features: the unchanged "
            f"{len(CORE_FEATURE_COLS)} engineered physiological "
            "CORE_FEATURE_COLS. Event probabilities are arithmetic means "
            "of window probabilities."
        ),
        (
            "Standard LOSO excludes the held-out participant. The "
            "subject-and-task-disjoint design additionally excludes the "
            "held-out exact task identity from every training "
            "participant. Task identity, protocol order, context "
            "features, CNN, video, and audio are not inputs."
        ),
        (
            "Training uses participant-task equal window weights "
            "(1 / windows in participant-task), with imputation and "
            "scaling fit only on the corresponding training records. "
            "Model hyperparameters are fixed and unchanged."
        ),
        "",
        "## Exact Ns",
        "",
        (
            f"Each complete model-design comparison contains "
            f"{int(primary['n_participants'].min())} participants, "
            f"{int(primary['n_participant_task_events'].min())} "
            "participant-task events, "
            f"{int(primary['n_unique_heldout_tasks'].min())} exact task "
            "identities, and "
            f"{int(primary['n_windows'].min())} constituent windows."
        ),
        "",
        "## Primary Results",
        "",
        (
            "| Model | Standard LOSO BA (95% CI) | Standard LOSO "
            "macro-F1 (95% CI) | Subject-task-disjoint BA (95% CI) | "
            "Subject-task-disjoint macro-F1 (95% CI) |"
        ),
        "|---|---:|---:|---:|---:|",
        *pivot_rows,
        "",
        "## Paired Differences",
        "",
        (
            "Differences are subject-and-task-disjoint minus standard "
            "LOSO on identical participant-task events, with 2,000 "
            "paired participant-cluster bootstrap resamples."
        ),
        "",
        (
            "| Model | Metric | Difference | 95% CI | Participants | "
            "Events |"
        ),
        "|---|---|---:|---:|---:|---:|",
        *paired_rows,
        "",
        "## Model Consistency",
        "",
        consistency_sentence,
        (
            " Subject-and-task-disjoint balanced accuracy estimates "
            f"(95% CIs): {exact_estimates}."
        ),
        "",
        "## Diagnostics",
        "",
        (
            f"Failed or unevaluable model-event folds: {len(failed)}. "
            f"Audit rows with a single training class: "
            f"{len(single_class_train)}. Completed disjoint event folds "
            f"whose window predictions used one class only: "
            f"{len(disjoint_collapsed)}."
        ),
        (
            "Every exact task is class-constant under the protocol "
            "mapping, so task-specific balanced accuracy is undefined. "
            "The affected tasks are: "
            + ", ".join(one_class_tasks.astype(str))
            + ". Task rows are descriptive only."
        ),
        "",
        "## Limitations",
        "",
        (
            "Exact task identity is confounded with protocol content and "
            "order. Removing a task changes the training distribution, "
            "and task-specific diagnostics contain only one target "
            "class. The estimates therefore characterize transfer to "
            "unrepresented protocol tasks; they do not isolate a causal "
            "task effect."
        ),
        "",
        "## Validation",
        "",
        (
            f"Automated checks passed: "
            f"{int(checks['passed'].sum())}/{len(checks)}."
        ),
        (
            "Prior-output manifest digest before: "
            f"`{manifest_before_digest}`"
        ),
        (
            "Prior-output manifest digest after: "
            f"`{manifest_after_digest}`"
        ),
        (
            "The matching digests cover every file that existed under "
            "`paper_results_v2/` before this addendum, excluding only "
            "this new output directory."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reproduction_commands(path: Path) -> None:
    commands = [
        (
            "python paper_results_v2/scripts/"
            "subject_task_disjoint_all_models.py "
            "--out-dir paper_results_v2/"
            "reproduced_subject_task_disjoint_all_tree_models "
            "--bootstrap 2000"
        ),
        (
            "python paper_results_v2/scripts/"
            "validate_subject_task_disjoint_all_models.py "
            "--out-dir paper_results_v2/"
            "reproduced_subject_task_disjoint_all_tree_models"
        ),
    ]
    path.write_text("\n".join(commands) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run StressID standard-LOSO and subject-and-task-disjoint "
            "participant-task evaluation for fixed tree models."
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "stressid_run" / "stressid_features.csv",
    )
    parser.add_argument(
        "--retained-predictions",
        type=Path,
        default=ROOT / "paper_results_v2" / "stressid_predictions.parquet",
    )
    parser.add_argument(
        "--prior-rf-predictions",
        type=Path,
        default=(
            ROOT
            / "paper_results_v2"
            / "addendum_context_incremental_value"
            / "subject_task_disjoint_predictions.parquet"
        ),
    )
    parser.add_argument(
        "--source-dir", type=Path, default=ROOT / "paper_results_v2",
        help="Directory to check for unintended modifications during analysis.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            ROOT
            / "paper_results_v2"
            / "addendum_subject_task_disjoint_all_tree_models"
        ),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=BOOTSTRAP_N,
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODELS),
    )
    parser.add_argument(
        "--event-limit",
        type=int,
        default=None,
        help="Smoke-test only; omitted for the complete analysis.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    source_dir = args.source_dir.resolve()
    models = [
        value.strip()
        for value in args.models.split(",")
        if value.strip()
    ]
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        raise ValueError(f"unsupported models: {unknown}")
    if models != MODELS and args.event_limit is None:
        raise ValueError(
            "complete analysis requires RandomForest, XGBoost, LightGBM"
        )
    output_guard(args.out_dir)
    prior_manifest = snapshot_prior_files(source_dir, args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.out_dir / "run.log")
    write_csv(
        prior_manifest,
        args.out_dir / "prior_outputs_manifest.csv",
        logger,
    )
    manifest_before_digest = sha256_manifest(
        prior_manifest,
        "sha256_before",
    )
    logger(
        f"python={platform.python_version()}; seed={RANDOM_STATE}; "
        f"bootstrap={args.bootstrap}"
    )
    seed_everything(RANDOM_STATE)

    windows = prepare_windows(args.features.resolve())
    logger(
        f"eligible data: {windows['subject_id'].nunique()} subjects, "
        f"{windows['event_id'].nunique()} events, {len(windows)} windows, "
        f"{windows['task_id'].nunique()} tasks"
    )
    standard_predictions, standard_audit, standard_checks = (
        make_standard_predictions(
            windows,
            args.retained_predictions.resolve(),
            models,
            logger,
        )
    )
    disjoint_predictions, disjoint_audit, disjoint_checks = (
        make_disjoint_predictions(
            windows,
            models,
            logger,
            args.event_limit,
        )
    )
    predictions = pd.concat(
        [standard_predictions, disjoint_predictions],
        ignore_index=True,
    )
    predictions = predictions.sort_values(
        [
            "model",
            "evaluation_design",
            "prediction_level",
            "subject_id",
            "task_id",
            "start_sec",
            "window_id",
        ],
        na_position="last",
    ).reset_index(drop=True)
    predictions.to_parquet(
        args.out_dir / "subject_task_disjoint_predictions.parquet",
        index=False,
    )
    logger(
        "wrote subject_task_disjoint_predictions.parquet "
        f"({len(predictions)} rows)"
    )
    audit = pd.concat(
        [standard_audit, disjoint_audit],
        ignore_index=True,
        sort=False,
    ).sort_values(
        ["model", "evaluation_design", "test_subject_id", "test_task_id"]
    )
    write_csv(
        audit,
        args.out_dir / "subject_task_disjoint_fold_audit.csv",
        logger,
    )

    event_predictions = predictions[
        predictions["prediction_level"] == "participant_task"
    ].copy()
    performance = summarize_performance(
        event_predictions,
        args.bootstrap,
    )
    paired = paired_differences(
        event_predictions,
        args.bootstrap,
    )
    task_table = task_diagnostics(event_predictions)
    write_csv(
        performance,
        args.out_dir / "subject_task_disjoint_all_models.csv",
        logger,
    )
    write_csv(
        paired,
        args.out_dir
        / "subject_task_disjoint_paired_differences.csv",
        logger,
    )
    write_csv(
        task_table,
        args.out_dir / "subject_task_disjoint_task_diagnostics.csv",
        logger,
    )
    write_json(
        make_numbers(performance, paired),
        args.out_dir / "subject_task_disjoint_numbers.json",
    )
    logger("wrote subject_task_disjoint_numbers.json")
    write_reproduction_commands(
        args.out_dir / "reproduction_commands.txt"
    )
    logger("wrote reproduction_commands.txt")

    preserved = verify_prior_files(source_dir, prior_manifest)
    write_csv(
        preserved,
        args.out_dir / "prior_outputs_preserved.csv",
        logger,
    )
    manifest_after_digest = sha256_manifest(
        preserved,
        "sha256_after",
    )
    checks = build_validation_checks(
        predictions,
        audit,
        standard_checks,
        disjoint_checks,
        args.prior_rf_predictions.resolve(),
        preserved,
    )
    write_csv(
        checks,
        args.out_dir / "validation_checks.csv",
        logger,
    )
    write_summary(
        args.out_dir / "subject_task_disjoint_summary.md",
        performance,
        paired,
        task_table,
        audit,
        checks,
        manifest_before_digest,
        manifest_after_digest,
    )
    logger("wrote subject_task_disjoint_summary.md")

    failed_checks = checks[~checks["passed"]]
    if len(failed_checks):
        logger(
            f"VALIDATION FAILED: {len(failed_checks)}/{len(checks)} "
            "checks failed"
        )
        print(failed_checks.to_string(index=False), flush=True)
        return 2
    logger(f"VALIDATION PASSED: {len(checks)}/{len(checks)} checks")
    headline = performance[
        performance["metric"].isin(
            ["balanced_accuracy", "macro_f1"]
        )
    ][
        [
            "model",
            "evaluation_design",
            "metric",
            "estimate",
            "ci_low",
            "ci_high",
            "n_participants",
            "n_participant_task_events",
        ]
    ]
    print("\nHEADLINE RESULTS", flush=True)
    print(headline.to_string(index=False), flush=True)
    print("\nPAIRED DIFFERENCES", flush=True)
    print(paired.to_string(index=False), flush=True)
    print(
        "\nPRESERVATION "
        f"{int(preserved['unchanged'].sum())}/{len(preserved)} "
        f"digest={manifest_after_digest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
