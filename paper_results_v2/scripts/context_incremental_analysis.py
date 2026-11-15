from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_common import (  # noqa: E402
    RANDOM_STATE,
    RunLogger,
    ensure_no_subject_leakage,
    event_equal_weights,
    robust_impute_scale,
    seed_everything,
    stable_seed,
)
from stressid_pipeline import CORE_FEATURE_COLS, TASKS, make_model  # noqa: E402


warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBM.* was fitted with feature names",
)


try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


MODELS = ["RandomForest", "XGBoost", "LightGBM"]
CONTINUOUS_TARGETS = ["stress", "arousal"]
BINARY_TARGET = "stress_binary_dataset"
BINARY_COLUMN = "binary_stress_official"
VIDEO_TASKS = {"Video1", "Video2"}
TASK_ORDER = {task: index + 1 for index, task in enumerate(TASKS)}
OUTPUT_FILES = [
    "context_incremental_results.csv",
    "residual_prediction_results.csv",
    "variance_decomposition.csv",
    "uncertainty_by_unit.csv",
    "subject_task_disjoint_results.csv",
    "context_incremental_summary.md",
    "context_incremental_numbers.json",
    "context_incremental_event_predictions.parquet",
    "residual_event_predictions.parquet",
    "fold_diagnostics.csv",
    "prior_outputs_manifest.csv",
    "prior_outputs_preserved.csv",
    "run.log",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(path: Path) -> str:
    return sha256_file(path)


def slug(value: object) -> str:
    text = str(value).strip().lower()
    return "".join(char if char.isalnum() else "_" for char in text).strip("_")


def guard_new_outputs(out_dir: Path) -> None:
    existing = [name for name in OUTPUT_FILES if (out_dir / name).exists()]
    figure_dir = out_dir / "figures"
    if figure_dir.exists() and any(figure_dir.iterdir()):
        existing.append("figures/")
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing addendum outputs: " + ", ".join(existing)
        )


def snapshot_prior_outputs(source_dir: Path, out_dir: Path) -> pd.DataFrame:
    excluded_roots = {
        (source_dir / "scripts").resolve(),
        (source_dir / "tmp").resolve(),
        out_dir.resolve(),
    }
    rows: list[dict[str, object]] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if any(root == resolved or root in resolved.parents for root in excluded_roots):
            continue
        rows.append(
            {
                "path": str(path.relative_to(source_dir)).replace("\\", "/"),
                "size": int(path.stat().st_size),
                "sha256_before": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def verify_prior_outputs(
    source_dir: Path,
    out_dir: Path,
    before: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for record in before.to_dict("records"):
        path = source_dir / str(record["path"])
        exists = path.exists()
        after_hash = sha256_file(path) if exists else ""
        rows.append(
            {
                **record,
                "exists_after": bool(exists),
                "sha256_after": after_hash,
                "unchanged": bool(exists and after_hash == record["sha256_before"]),
            }
        )
    return pd.DataFrame(rows)


def fixed_binary_metrics(
    truth: np.ndarray,
    pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(truth, dtype=int)
    y_pred = np.asarray(pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
    supports = cm.sum(axis=1)
    recalls = np.divide(
        np.diag(cm),
        supports,
        out=np.zeros(2, dtype=float),
        where=supports > 0,
    )
    return {
        "balanced_accuracy": float(recalls.mean()),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
        ),
    }


def binary_metrics_from_confusions(
    cms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    diagonal = np.diagonal(cms, axis1=-2, axis2=-1)
    true_support = cms.sum(axis=-1)
    predicted_support = cms.sum(axis=-2)
    recalls = np.divide(
        diagonal,
        true_support,
        out=np.zeros_like(diagonal, dtype=float),
        where=true_support > 0,
    )
    denominator = (
        2.0 * diagonal
        + (predicted_support - diagonal)
        + (true_support - diagonal)
    )
    f1_values = np.divide(
        2.0 * diagonal,
        denominator,
        out=np.zeros_like(diagonal, dtype=float),
        where=denominator > 0,
    )
    return recalls.mean(axis=-1), f1_values.mean(axis=-1)


def regression_metrics(
    truth: np.ndarray,
    pred: np.ndarray,
    include_r2: bool = True,
) -> dict[str, float]:
    y_true = np.asarray(truth, dtype=float)
    y_pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if not len(y_true):
        return {
            "spearman_rho": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "r_squared": np.nan,
        }
    rho = (
        spearmanr(y_true, y_pred).statistic
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1
        else np.nan
    )
    squared_error = np.square(y_true - y_pred)
    centered = np.square(y_true - y_true.mean()).sum()
    r_squared = (
        float(1.0 - squared_error.sum() / centered)
        if include_r2 and centered > 0
        else np.nan
    )
    return {
        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean(squared_error))),
        "r_squared": r_squared,
    }


def metric_value(
    frame: pd.DataFrame,
    metric: str,
    truth_col: str = "y_true",
    pred_col: str = "y_pred",
) -> float:
    if metric in {"balanced_accuracy", "macro_f1"}:
        return fixed_binary_metrics(
            frame[truth_col].to_numpy(),
            frame[pred_col].to_numpy(),
        )[metric]
    return regression_metrics(
        frame[truth_col].to_numpy(),
        frame[pred_col].to_numpy(),
        include_r2=True,
    )[metric]


def subject_bootstrap_ci(
    frame: pd.DataFrame,
    metric: str,
    n_boot: int,
    seed: int,
    truth_col: str = "y_true",
    pred_col: str = "y_pred",
) -> tuple[float, float, int]:
    groups = [
        (
            group[truth_col].to_numpy(),
            group[pred_col].to_numpy(),
        )
        for _, group in frame.groupby("subject_id", sort=True)
    ]
    if not groups:
        return np.nan, np.nan, 0
    if metric == "spearman_rho":
        all_truth = np.concatenate([group[0] for group in groups])
        all_pred = np.concatenate([group[1] for group in groups])
        if len(np.unique(all_truth)) < 2 or len(np.unique(all_pred)) < 2:
            return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    selected = rng.integers(
        0,
        len(groups),
        size=(n_boot, len(groups)),
    )
    selection_counts = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=len(groups)),
        1,
        selected,
    )
    if metric in {"balanced_accuracy", "macro_f1"}:
        matrices = np.stack(
            [
                confusion_matrix(
                    truth.astype(int),
                    pred.astype(int),
                    labels=[0, 1],
                ).astype(float)
                for truth, pred in groups
            ],
            axis=0,
        )
        sampled = np.einsum(
            "bs,sij->bij",
            selection_counts,
            matrices,
            optimize=True,
        )
        balanced_accuracy, macro_f1 = binary_metrics_from_confusions(
            sampled
        )
        values = (
            balanced_accuracy
            if metric == "balanced_accuracy"
            else macro_f1
        )
    elif metric in {"mae", "rmse", "r_squared"}:
        counts = np.asarray(
            [len(truth) for truth, _ in groups],
            dtype=float,
        )
        absolute_error = np.asarray(
            [
                np.abs(truth.astype(float) - pred.astype(float)).sum()
                for truth, pred in groups
            ],
            dtype=float,
        )
        squared_error = np.asarray(
            [
                np.square(
                    truth.astype(float) - pred.astype(float)
                ).sum()
                for truth, pred in groups
            ],
            dtype=float,
        )
        total_n = selection_counts @ counts
        if metric == "mae":
            values = (selection_counts @ absolute_error) / total_n
        elif metric == "rmse":
            values = np.sqrt(
                (selection_counts @ squared_error) / total_n
            )
        else:
            truth_sum = np.asarray(
                [truth.astype(float).sum() for truth, _ in groups]
            )
            truth_square_sum = np.asarray(
                [
                    np.square(truth.astype(float)).sum()
                    for truth, _ in groups
                ]
            )
            sampled_sum = selection_counts @ truth_sum
            sampled_square_sum = selection_counts @ truth_square_sum
            total_variation = (
                sampled_square_sum
                - np.square(sampled_sum) / total_n
            )
            sampled_error = selection_counts @ squared_error
            values = np.full(n_boot, np.nan, dtype=float)
            valid = total_variation > 0
            values[valid] = (
                1.0
                - sampled_error[valid] / total_variation[valid]
            )
    elif metric == "spearman_rho":
        values = np.full(n_boot, np.nan, dtype=float)
        for index, sample in enumerate(selected):
            truth = np.concatenate([groups[item][0] for item in sample])
            pred = np.concatenate([groups[item][1] for item in sample])
            values[index] = spearmanr(truth, pred).statistic
    else:
        raise ValueError(metric)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.nan, np.nan, 0
    return (
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
        int(len(finite)),
    )


def paired_subject_bootstrap_ci(
    frame: pd.DataFrame,
    metric: str,
    left_col: str,
    right_col: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, int]:
    groups = [
        (
            group["y_true"].to_numpy(),
            group[left_col].to_numpy(),
            group[right_col].to_numpy(),
        )
        for _, group in frame.groupby("subject_id", sort=True)
    ]
    if not groups:
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    selected = rng.integers(
        0,
        len(groups),
        size=(n_boot, len(groups)),
    )
    selection_counts = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=len(groups)),
        1,
        selected,
    )
    if metric in {"balanced_accuracy", "macro_f1"}:
        left_matrices = np.stack(
            [
                confusion_matrix(
                    truth.astype(int),
                    left.astype(int),
                    labels=[0, 1],
                ).astype(float)
                for truth, left, _ in groups
            ],
            axis=0,
        )
        right_matrices = np.stack(
            [
                confusion_matrix(
                    truth.astype(int),
                    right.astype(int),
                    labels=[0, 1],
                ).astype(float)
                for truth, _, right in groups
            ],
            axis=0,
        )
        sampled_left = np.einsum(
            "bs,sij->bij",
            selection_counts,
            left_matrices,
            optimize=True,
        )
        sampled_right = np.einsum(
            "bs,sij->bij",
            selection_counts,
            right_matrices,
            optimize=True,
        )
        left_ba, left_f1 = binary_metrics_from_confusions(
            sampled_left
        )
        right_ba, right_f1 = binary_metrics_from_confusions(
            sampled_right
        )
        values = (
            left_ba - right_ba
            if metric == "balanced_accuracy"
            else left_f1 - right_f1
        )
    elif metric in {"mae", "rmse", "r_squared"}:
        counts = np.asarray(
            [len(truth) for truth, _, _ in groups],
            dtype=float,
        )
        total_n = selection_counts @ counts
        truth_sum = np.asarray(
            [truth.astype(float).sum() for truth, _, _ in groups]
        )
        truth_square_sum = np.asarray(
            [
                np.square(truth.astype(float)).sum()
                for truth, _, _ in groups
            ]
        )

        def error_vectors(prediction_index: int) -> tuple[np.ndarray, np.ndarray]:
            absolute = np.asarray(
                [
                    np.abs(
                        group[0].astype(float)
                        - group[prediction_index].astype(float)
                    ).sum()
                    for group in groups
                ]
            )
            squared = np.asarray(
                [
                    np.square(
                        group[0].astype(float)
                        - group[prediction_index].astype(float)
                    ).sum()
                    for group in groups
                ]
            )
            return absolute, squared

        left_absolute, left_squared = error_vectors(1)
        right_absolute, right_squared = error_vectors(2)
        if metric == "mae":
            left_values = (
                selection_counts @ left_absolute
            ) / total_n
            right_values = (
                selection_counts @ right_absolute
            ) / total_n
        elif metric == "rmse":
            left_values = np.sqrt(
                (selection_counts @ left_squared) / total_n
            )
            right_values = np.sqrt(
                (selection_counts @ right_squared) / total_n
            )
        else:
            sampled_sum = selection_counts @ truth_sum
            sampled_square_sum = selection_counts @ truth_square_sum
            total_variation = (
                sampled_square_sum
                - np.square(sampled_sum) / total_n
            )
            left_error = selection_counts @ left_squared
            right_error = selection_counts @ right_squared
            left_values = np.divide(
                left_error,
                total_variation,
                out=np.full(n_boot, np.nan),
                where=total_variation > 0,
            )
            right_values = np.divide(
                right_error,
                total_variation,
                out=np.full(n_boot, np.nan),
                where=total_variation > 0,
            )
            left_values = 1.0 - left_values
            right_values = 1.0 - right_values
        values = left_values - right_values
    elif metric == "spearman_rho":
        all_truth = np.concatenate([group[0] for group in groups])
        all_left = np.concatenate([group[1] for group in groups])
        all_right = np.concatenate([group[2] for group in groups])
        if (
            len(np.unique(all_truth)) < 2
            or len(np.unique(all_left)) < 2
            or len(np.unique(all_right)) < 2
        ):
            return np.nan, np.nan, 0
        values = np.full(n_boot, np.nan, dtype=float)
        for index, sample in enumerate(selected):
            truth = np.concatenate([groups[item][0] for item in sample])
            left = np.concatenate([groups[item][1] for item in sample])
            right = np.concatenate([groups[item][2] for item in sample])
            values[index] = (
                spearmanr(truth, left).statistic
                - spearmanr(truth, right).statistic
            )
    else:
        raise ValueError(metric)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.nan, np.nan, 0
    return (
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
        int(len(finite)),
    )


def make_regressor(model_name: str):
    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=180,
            max_depth=8,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_name == "XGBoost":
        if XGBRegressor is None:
            raise RuntimeError("XGBoost regressor unavailable")
        return XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
    if model_name == "LightGBM":
        if LGBMRegressor is None:
            raise RuntimeError("LightGBM regressor unavailable")
        return LGBMRegressor(
            objective="regression",
            n_estimators=160,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
        )
    raise ValueError(model_name)


def encode_training_tasks(
    train_tasks: pd.Series,
    test_tasks: pd.Series,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    present = set(train_tasks.astype(str))
    categories = [task for task in TASKS if task in present]
    if not categories:
        raise RuntimeError("no task categories in outer-training fold")
    train = np.column_stack(
        [(train_tasks.astype(str) == task).to_numpy(dtype=float) for task in categories]
    )
    test = np.column_stack(
        [(test_tasks.astype(str) == task).to_numpy(dtype=float) for task in categories]
    )
    return train, test, categories


def event_table(
    windows: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    rows = []
    for (subject, task), group in windows.groupby(
        ["subject_id", "task_id"],
        sort=True,
    ):
        truth = group[target_col].dropna().unique()
        if len(truth) != 1:
            raise RuntimeError(
                f"{target_col} is not event-constant for {subject}|{task}: {truth}"
            )
        rows.append(
            {
                "subject_id": str(subject),
                "task_id": str(task),
                "event_id": f"{subject}|{task}",
                "y_true": float(truth[0]),
                "n_windows": int(len(group)),
                "protocol_order_index": int(TASK_ORDER[str(task)]),
            }
        )
    return pd.DataFrame(rows)


def append_event_predictions(
    rows: list[dict[str, object]],
    test_frame: pd.DataFrame,
    scores: np.ndarray,
    target: str,
    target_type: str,
    model_name: str,
    predictor_set: str,
    heldout: str,
) -> None:
    scored = test_frame[
        ["subject_id", "task_id", "event_id", "y_true"]
    ].copy()
    scored["score"] = np.asarray(scores, dtype=float)
    for (subject, task, event), group in scored.groupby(
        ["subject_id", "task_id", "event_id"],
        sort=True,
    ):
        truth = group["y_true"].unique()
        if len(truth) != 1:
            raise RuntimeError(f"nonconstant event truth: {event}")
        score = float(group["score"].mean())
        rows.append(
            {
                "analysis_class": "prespecified_primary",
                "target": target,
                "target_type": target_type,
                "model": model_name,
                "predictor_set": predictor_set,
                "subject_id": str(subject),
                "task_id": str(task),
                "event_id": str(event),
                "fold_subject": str(heldout),
                "y_true": float(truth[0]),
                "y_score": score,
                "y_pred": (
                    int(score >= 0.5)
                    if target_type == "binary"
                    else score
                ),
                "n_windows": int(len(group)),
                "aggregation": "mean_window_prediction",
            }
        )


def run_context_incremental(
    features: pd.DataFrame,
    models: list[str],
    n_boot: int,
    logger: RunLogger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    target_specs = [
        ("stress", "stress", "continuous"),
        ("arousal", "arousal", "continuous"),
        (BINARY_TARGET, BINARY_COLUMN, "binary"),
    ]
    for target, target_col, target_type in target_specs:
        windows = features.loc[features[target_col].notna()].copy()
        windows["subject_id"] = windows["subject_id"].astype(str)
        windows["task_id"] = windows["task_id"].astype(str)
        windows["event_id"] = (
            windows["subject_id"] + "|" + windows["task_id"]
        )
        windows["y_true"] = windows[target_col].astype(float)
        events = event_table(windows, target_col)
        subjects = windows["subject_id"].to_numpy(dtype=str)
        unique_subjects = sorted(windows["subject_id"].unique())
        logger(
            f"incremental target={target}: {len(windows)} windows, "
            f"{len(events)} events, {len(unique_subjects)} subjects"
        )
        for fold_number, heldout in enumerate(unique_subjects, start=1):
            train_idx = np.flatnonzero(subjects != heldout)
            test_idx = np.flatnonzero(subjects == heldout)
            ensure_no_subject_leakage(subjects, train_idx, test_idx)
            train_windows = windows.iloc[train_idx].copy()
            test_windows = windows.iloc[test_idx].copy()
            train_events = events.loc[
                events["subject_id"] != heldout
            ].copy()
            test_events = events.loc[
                events["subject_id"] == heldout
            ].copy()
            task_context = train_events.groupby("task_id")["y_true"].mean()
            fallback = float(train_events["y_true"].mean())
            missing_context_tasks = sorted(
                set(test_events["task_id"]) - set(task_context.index)
            )
            for row in test_events.to_dict("records"):
                score = float(task_context.get(row["task_id"], fallback))
                if target_type == "binary":
                    context_model = "ContextTaskPrevalence"
                else:
                    context_model = "ContextTaskMean"
                predictions.append(
                    {
                        "analysis_class": "prespecified_primary",
                        "target": target,
                        "target_type": target_type,
                        "model": context_model,
                        "predictor_set": "context_only",
                        "subject_id": row["subject_id"],
                        "task_id": row["task_id"],
                        "event_id": row["event_id"],
                        "fold_subject": heldout,
                        "y_true": float(row["y_true"]),
                        "y_score": score,
                        "y_pred": (
                            int(score >= 0.5)
                            if target_type == "binary"
                            else score
                        ),
                        "n_windows": int(row["n_windows"]),
                        "aggregation": "training_subject_task_statistic",
                    }
                )

            order_train = train_events[
                ["protocol_order_index"]
            ].to_numpy(dtype=float)
            order_test = test_events[
                ["protocol_order_index"]
            ].to_numpy(dtype=float)
            if target_type == "binary":
                order_model = LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                )
                order_model.fit(
                    order_train,
                    train_events["y_true"].to_numpy(dtype=int),
                )
                order_scores = order_model.predict_proba(order_test)[:, 1]
                order_model_name = "ProtocolOrderLogistic"
            else:
                order_model = LinearRegression()
                order_model.fit(
                    order_train,
                    train_events["y_true"].to_numpy(dtype=float),
                )
                order_scores = order_model.predict(order_test)
                order_model_name = "ProtocolOrderLinear"
            for row, score in zip(
                test_events.to_dict("records"),
                order_scores,
            ):
                predictions.append(
                    {
                        "analysis_class": "prespecified_primary",
                        "target": target,
                        "target_type": target_type,
                        "model": order_model_name,
                        "predictor_set": "protocol_order_only",
                        "subject_id": row["subject_id"],
                        "task_id": row["task_id"],
                        "event_id": row["event_id"],
                        "fold_subject": heldout,
                        "y_true": float(row["y_true"]),
                        "y_score": float(score),
                        "y_pred": (
                            int(float(score) >= 0.5)
                            if target_type == "binary"
                            else float(score)
                        ),
                        "n_windows": int(row["n_windows"]),
                        "aggregation": "training_fold_order_model",
                    }
                )

            raw = windows[CORE_FEATURE_COLS].to_numpy(dtype=float)
            (
                x_train,
                x_test,
                medians,
                means,
                scales,
            ) = robust_impute_scale(raw[train_idx], raw[test_idx])
            task_train, task_test, categories = encode_training_tasks(
                train_windows["task_id"],
                test_windows["task_id"],
            )
            combined_train = np.column_stack([x_train, task_train])
            combined_test = np.column_stack([x_test, task_test])
            weights = event_equal_weights(
                train_windows,
                ["subject_id", "task_id"],
            )
            y_train = train_windows["y_true"].to_numpy(
                dtype=int if target_type == "binary" else float
            )
            for model_name in models:
                try:
                    for predictor_set, train_matrix, test_matrix in [
                        ("physiology_only", x_train, x_test),
                        (
                            "context_plus_physiology",
                            combined_train,
                            combined_test,
                        ),
                    ]:
                        model = (
                            make_model(model_name)
                            if target_type == "binary"
                            else make_regressor(model_name)
                        )
                        if model is None:
                            raise RuntimeError(f"{model_name} unavailable")
                        model.fit(
                            train_matrix,
                            y_train,
                            sample_weight=weights,
                        )
                        if target_type == "binary":
                            classes = list(model.classes_)
                            scores = model.predict_proba(test_matrix)[
                                :, classes.index(1)
                            ]
                        else:
                            scores = model.predict(test_matrix)
                        append_event_predictions(
                            predictions,
                            test_windows,
                            scores,
                            target,
                            target_type,
                            model_name,
                            predictor_set,
                            heldout,
                        )
                    diagnostics.append(
                        {
                            "analysis": "context_incremental",
                            "target": target,
                            "model": model_name,
                            "fold_subject": heldout,
                            "n_train_subjects": int(
                                train_windows["subject_id"].nunique()
                            ),
                            "n_test_subjects": int(
                                test_windows["subject_id"].nunique()
                            ),
                            "n_train_events": int(
                                train_windows["event_id"].nunique()
                            ),
                            "n_test_events": int(
                                test_windows["event_id"].nunique()
                            ),
                            "subject_overlap": 0,
                            "context_missing_test_tasks": "|".join(
                                missing_context_tasks
                            ),
                            "task_encoder_categories": "|".join(categories),
                            "task_encoder_fit_subjects": int(
                                train_windows["subject_id"].nunique()
                            ),
                            "imputer_fit_subjects": int(
                                train_windows["subject_id"].nunique()
                            ),
                            "scaler_fit_subjects": int(
                                train_windows["subject_id"].nunique()
                            ),
                            "sample_weight_min": float(weights.min()),
                            "sample_weight_max": float(weights.max()),
                            "imputer_hash": hashlib.sha256(
                                medians.tobytes()
                            ).hexdigest(),
                            "scaler_mean_hash": hashlib.sha256(
                                means.tobytes()
                            ).hexdigest(),
                            "scaler_scale_hash": hashlib.sha256(
                                scales.tobytes()
                            ).hexdigest(),
                        }
                    )
                except Exception as exc:
                    logger.exception(
                        f"incremental {target} {model_name} {heldout}",
                        exc,
                    )
                    raise
            if fold_number == 1 or fold_number % 10 == 0:
                logger(
                    f"incremental {target} fold "
                    f"{fold_number}/{len(unique_subjects)}"
                )
    event_predictions = pd.DataFrame(predictions)
    results = summarize_incremental_predictions(
        event_predictions,
        models,
        n_boot,
    )
    return event_predictions, results, pd.DataFrame(diagnostics)


def summarize_incremental_predictions(
    predictions: pd.DataFrame,
    models: list[str],
    n_boot: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (target, target_type, model, predictor_set), group in predictions.groupby(
        ["target", "target_type", "model", "predictor_set"],
        sort=True,
    ):
        metrics = (
            ["balanced_accuracy", "macro_f1"]
            if target_type == "binary"
            else ["spearman_rho", "mae", "rmse"]
        )
        for metric in metrics:
            estimate = metric_value(group, metric)
            low, high, valid = subject_bootstrap_ci(
                group,
                metric,
                n_boot,
                stable_seed("context", target, model, predictor_set, metric),
            )
            rows.append(
                {
                    "analysis_class": "prespecified_primary",
                    "row_type": "performance",
                    "target": target,
                    "target_type": target_type,
                    "model": model,
                    "predictor_set": predictor_set,
                    "comparison": "",
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_bootstrap": n_boot,
                    "valid_bootstrap_resamples": valid,
                    "bootstrap_unit": "participant",
                    "n_subjects": int(group["subject_id"].nunique()),
                    "n_events": int(len(group)),
                    "n_windows": int(group["n_windows"].sum()),
                    "outer_validation": "strict_LOSO",
                    "evaluation_unit": "participant_task",
                    "aggregation": str(group["aggregation"].iloc[0]),
                    "difference_direction": "",
                    "deployment_status": (
                        "diagnostic_not_deployment_feature"
                        if "context" in predictor_set
                        or predictor_set == "protocol_order_only"
                        else "physiology_feature_model"
                    ),
                }
            )

    comparison_specs = [
        (
            "combined_minus_context_only",
            "context_plus_physiology",
            "context_only",
        ),
        (
            "physiology_only_minus_context_only",
            "physiology_only",
            "context_only",
        ),
        (
            "combined_minus_physiology_only",
            "context_plus_physiology",
            "physiology_only",
        ),
    ]
    for target in sorted(predictions["target"].unique()):
        target_df = predictions[predictions["target"] == target]
        target_type = str(target_df["target_type"].iloc[0])
        context = target_df[
            target_df["predictor_set"] == "context_only"
        ][["subject_id", "task_id", "event_id", "y_true", "y_pred"]].rename(
            columns={"y_pred": "context_only"}
        )
        metrics = (
            ["balanced_accuracy", "macro_f1"]
            if target_type == "binary"
            else ["spearman_rho", "mae", "rmse"]
        )
        for model in models:
            model_df = target_df[target_df["model"] == model]
            physiology = model_df[
                model_df["predictor_set"] == "physiology_only"
            ][["subject_id", "task_id", "event_id", "y_pred"]].rename(
                columns={"y_pred": "physiology_only"}
            )
            combined = model_df[
                model_df["predictor_set"]
                == "context_plus_physiology"
            ][["subject_id", "task_id", "event_id", "y_pred"]].rename(
                columns={"y_pred": "context_plus_physiology"}
            )
            merged = (
                context.merge(
                    physiology,
                    on=["subject_id", "task_id", "event_id"],
                    how="inner",
                )
                .merge(
                    combined,
                    on=["subject_id", "task_id", "event_id"],
                    how="inner",
                )
            )
            if len(merged) != len(context):
                raise RuntimeError(
                    f"paired event mismatch for {target}/{model}: "
                    f"{len(merged)} vs {len(context)}"
                )
            for comparison, left_col, right_col in comparison_specs:
                for metric in metrics:
                    left = metric_value(
                        merged,
                        metric,
                        pred_col=left_col,
                    )
                    right = metric_value(
                        merged,
                        metric,
                        pred_col=right_col,
                    )
                    low, high, valid = paired_subject_bootstrap_ci(
                        merged,
                        metric,
                        left_col,
                        right_col,
                        n_boot,
                        stable_seed(
                            "context_difference",
                            target,
                            model,
                            comparison,
                            metric,
                        ),
                    )
                    rows.append(
                        {
                            "analysis_class": "prespecified_primary",
                            "row_type": "paired_difference",
                            "target": target,
                            "target_type": target_type,
                            "model": model,
                            "predictor_set": "",
                            "comparison": comparison,
                            "metric": metric,
                            "estimate": left - right,
                            "ci_low": low,
                            "ci_high": high,
                            "n_bootstrap": n_boot,
                            "valid_bootstrap_resamples": valid,
                            "bootstrap_unit": "participant",
                            "n_subjects": int(
                                merged["subject_id"].nunique()
                            ),
                            "n_events": int(len(merged)),
                            "n_windows": int(
                                target_df[
                                    target_df["model"] == model
                                ]["n_windows"].sum()
                                / 2
                            ),
                            "outer_validation": "strict_LOSO",
                            "evaluation_unit": "participant_task",
                            "aggregation": "mean_window_prediction",
                            "difference_direction": "left_minus_right",
                            "deployment_status": (
                                "task_context_is_diagnostic_only"
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def run_residual_prediction(
    features: pd.DataFrame,
    models: list[str],
    n_boot: int,
    logger: RunLogger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for target in CONTINUOUS_TARGETS:
        windows = features.loc[features[target].notna()].copy()
        windows["subject_id"] = windows["subject_id"].astype(str)
        windows["task_id"] = windows["task_id"].astype(str)
        windows["event_id"] = (
            windows["subject_id"] + "|" + windows["task_id"]
        )
        events = event_table(windows, target).rename(
            columns={"y_true": "observed_rating"}
        )
        subjects = windows["subject_id"].to_numpy(dtype=str)
        unique_subjects = sorted(windows["subject_id"].unique())
        raw = windows[CORE_FEATURE_COLS].to_numpy(dtype=float)
        for fold_number, heldout in enumerate(unique_subjects, start=1):
            train_idx = np.flatnonzero(subjects != heldout)
            test_idx = np.flatnonzero(subjects == heldout)
            ensure_no_subject_leakage(subjects, train_idx, test_idx)
            train_windows = windows.iloc[train_idx].copy()
            test_windows = windows.iloc[test_idx].copy()
            train_events = events[
                events["subject_id"] != heldout
            ].copy()
            test_events = events[
                events["subject_id"] == heldout
            ].copy()
            task_means = train_events.groupby("task_id")[
                "observed_rating"
            ].mean()
            missing = sorted(
                set(test_events["task_id"]) - set(task_means.index)
            )
            if missing:
                raise RuntimeError(
                    f"residual test tasks absent from training: {missing}"
                )
            y_train_residual = (
                train_windows[target].to_numpy(dtype=float)
                - train_windows["task_id"].map(task_means).to_numpy(
                    dtype=float
                )
            )
            (
                x_train,
                x_test,
                medians,
                means,
                scales,
            ) = robust_impute_scale(raw[train_idx], raw[test_idx])
            weights = event_equal_weights(
                train_windows,
                ["subject_id", "task_id"],
            )
            for row in test_events.to_dict("records"):
                mean = float(task_means[row["task_id"]])
                residual = float(row["observed_rating"] - mean)
                prediction_rows.append(
                    {
                        "analysis_class": "prespecified_primary",
                        "target": target,
                        "model": "ZeroResidualBaseline",
                        "subject_id": row["subject_id"],
                        "task_id": row["task_id"],
                        "event_id": row["event_id"],
                        "fold_subject": heldout,
                        "observed_rating": float(
                            row["observed_rating"]
                        ),
                        "training_task_mean": mean,
                        "y_true": residual,
                        "y_pred": 0.0,
                        "n_windows": int(row["n_windows"]),
                        "aggregation": "fixed_zero",
                    }
                )
            for model_name in models:
                try:
                    model = make_regressor(model_name)
                    model.fit(
                        x_train,
                        y_train_residual,
                        sample_weight=weights,
                    )
                    scores = np.asarray(
                        model.predict(x_test),
                        dtype=float,
                    )
                    scored = test_windows[
                        ["subject_id", "task_id", "event_id"]
                    ].copy()
                    scored["score"] = scores
                    for (
                        subject,
                        task,
                        event,
                    ), group in scored.groupby(
                        ["subject_id", "task_id", "event_id"],
                        sort=True,
                    ):
                        observed = float(
                            test_windows.loc[
                                test_windows["event_id"] == event,
                                target,
                            ].iloc[0]
                        )
                        task_mean = float(task_means[task])
                        prediction_rows.append(
                            {
                                "analysis_class": (
                                    "prespecified_primary"
                                ),
                                "target": target,
                                "model": model_name,
                                "subject_id": str(subject),
                                "task_id": str(task),
                                "event_id": str(event),
                                "fold_subject": heldout,
                                "observed_rating": observed,
                                "training_task_mean": task_mean,
                                "y_true": observed - task_mean,
                                "y_pred": float(
                                    group["score"].mean()
                                ),
                                "n_windows": int(len(group)),
                                "aggregation": (
                                    "mean_window_prediction"
                                ),
                            }
                        )
                    diagnostics.append(
                        {
                            "analysis": "within_task_residual",
                            "target": target,
                            "model": model_name,
                            "fold_subject": heldout,
                            "n_train_subjects": int(
                                train_windows[
                                    "subject_id"
                                ].nunique()
                            ),
                            "n_test_subjects": 1,
                            "n_train_events": int(
                                train_windows["event_id"].nunique()
                            ),
                            "n_test_events": int(
                                test_windows["event_id"].nunique()
                            ),
                            "subject_overlap": 0,
                            "task_mean_fit_subjects": int(
                                train_events["subject_id"].nunique()
                            ),
                            "test_ratings_used_for_task_means": 0,
                            "missing_test_tasks": "|".join(missing),
                            "imputer_hash": hashlib.sha256(
                                medians.tobytes()
                            ).hexdigest(),
                            "scaler_mean_hash": hashlib.sha256(
                                means.tobytes()
                            ).hexdigest(),
                            "scaler_scale_hash": hashlib.sha256(
                                scales.tobytes()
                            ).hexdigest(),
                        }
                    )
                except Exception as exc:
                    logger.exception(
                        f"residual {target} {model_name} {heldout}",
                        exc,
                    )
                    raise
            if fold_number == 1 or fold_number % 10 == 0:
                logger(
                    f"residual {target} fold "
                    f"{fold_number}/{len(unique_subjects)}"
                )
    predictions = pd.DataFrame(prediction_rows)
    results = summarize_residual_predictions(
        predictions,
        n_boot,
    )
    return predictions, results, pd.DataFrame(diagnostics)


def summarize_residual_predictions(
    predictions: pd.DataFrame,
    n_boot: int,
) -> pd.DataFrame:
    rows = []
    for (target, model), group in predictions.groupby(
        ["target", "model"],
        sort=True,
    ):
        scopes = [("overall", "all_tasks", group)]
        scopes.extend(
            (
                "per_task_supplementary",
                str(task),
                task_group,
            )
            for task, task_group in group.groupby("task_id", sort=True)
        )
        for analysis_scope, task, scoped in scopes:
            for metric in [
                "spearman_rho",
                "mae",
                "rmse",
                "r_squared",
            ]:
                estimate = metric_value(scoped, metric)
                low, high, valid = subject_bootstrap_ci(
                    scoped,
                    metric,
                    n_boot,
                    stable_seed(
                        "residual",
                        target,
                        model,
                        task,
                        metric,
                    ),
                )
                rows.append(
                    {
                        "analysis_class": (
                            "prespecified_primary"
                            if analysis_scope == "overall"
                            else "supplementary_diagnostic"
                        ),
                        "analysis_scope": analysis_scope,
                        "target": target,
                        "task_id": task,
                        "model": model,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "n_bootstrap": n_boot,
                        "valid_bootstrap_resamples": valid,
                        "bootstrap_unit": "participant",
                        "n_subjects": int(
                            scoped["subject_id"].nunique()
                        ),
                        "n_events": int(len(scoped)),
                        "n_windows": int(scoped["n_windows"].sum()),
                        "outer_validation": "strict_LOSO",
                        "evaluation_unit": "participant_task_residual",
                        "task_mean_source": (
                            "outer_training_participants_only"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def run_variance_decomposition(
    features: pd.DataFrame,
    logger: RunLogger,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in CONTINUOUS_TARGETS:
        windows = features.loc[features[target].notna()].copy()
        events = event_table(windows, target).rename(
            columns={"y_true": "rating"}
        )
        events["task_id"] = pd.Categorical(
            events["task_id"],
            categories=[task for task in TASKS if task in set(events["task_id"])],
            ordered=True,
        )
        fit = None
        captured: list[str] = []
        for method in ["lbfgs", "powell"]:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = smf.mixedlm(
                        "rating ~ C(task_id)",
                        data=events,
                        groups=events["subject_id"],
                        re_formula="1",
                    )
                    fit = model.fit(
                        reml=True,
                        method=method,
                        maxiter=1000,
                        disp=False,
                    )
                    captured.extend(str(item.message) for item in caught)
                if bool(fit.converged):
                    break
            except Exception as exc:
                captured.append(f"{method}: {type(exc).__name__}: {exc}")
                fit = None
        if fit is None:
            rows.append(
                {
                    "analysis_class": "post_hoc_explanatory",
                    "row_type": "mixed_model",
                    "target": target,
                    "task_id": "",
                    "n_subjects": int(events["subject_id"].nunique()),
                    "n_events": int(len(events)),
                    "model": "rating ~ C(task) + (1|participant)",
                    "converged": False,
                    "warnings": " | ".join(captured),
                }
            )
            continue
        fixed_fitted = np.asarray(
            fit.model.exog @ np.asarray(fit.fe_params),
            dtype=float,
        )
        fixed_variance = float(np.var(fixed_fitted, ddof=0))
        participant_variance = float(
            np.asarray(fit.cov_re, dtype=float)[0, 0]
        )
        residual_variance = float(fit.scale)
        total = fixed_variance + participant_variance + residual_variance
        marginal = fixed_variance / total if total > 0 else np.nan
        conditional = (
            (fixed_variance + participant_variance) / total
            if total > 0
            else np.nan
        )
        rows.append(
            {
                "analysis_class": "post_hoc_explanatory",
                "row_type": "mixed_model",
                "target": target,
                "task_id": "",
                "n_subjects": int(events["subject_id"].nunique()),
                "n_events": int(len(events)),
                "model": "rating ~ C(task) + (1|participant)",
                "estimation": "REML",
                "converged": bool(fit.converged),
                "participant_random_intercept_variance": (
                    participant_variance
                ),
                "residual_variance": residual_variance,
                "fixed_task_context_variance": fixed_variance,
                "marginal_r_squared": marginal,
                "conditional_r_squared": conditional,
                "log_likelihood": float(fit.llf),
                "warnings": " | ".join(sorted(set(captured))),
                "interpretation_scope": (
                    "task_protocol_context_not_causal_task_effect"
                ),
            }
        )
        for task, group in events.groupby(
            "task_id",
            sort=False,
            observed=True,
        ):
            rating = group["rating"].to_numpy(dtype=float)
            rows.append(
                {
                    "analysis_class": "post_hoc_explanatory",
                    "row_type": "task_distribution",
                    "target": target,
                    "task_id": str(task),
                    "n_subjects": int(group["subject_id"].nunique()),
                    "n_events": int(len(group)),
                    "mean": float(np.mean(rating)),
                    "sd": float(np.std(rating, ddof=1)),
                    "median": float(np.median(rating)),
                    "minimum": float(np.min(rating)),
                    "maximum": float(np.max(rating)),
                    "q1": float(np.percentile(rating, 25)),
                    "q3": float(np.percentile(rating, 75)),
                    "interpretation_scope": (
                        "task_protocol_context_not_causal_task_effect"
                    ),
                }
            )
        logger(
            f"variance decomposition {target}: "
            f"marginal R2={marginal:.4f}, conditional R2={conditional:.4f}"
        )
    return pd.DataFrame(rows)


def ba_from_confusions(cms: np.ndarray) -> np.ndarray:
    diagonal = np.diagonal(cms, axis1=-2, axis2=-1)
    supports = cms.sum(axis=-1)
    recalls = np.divide(
        diagonal,
        supports,
        out=np.zeros_like(diagonal, dtype=float),
        where=supports > 0,
    )
    return recalls.mean(axis=-1)


def bootstrap_binary_by_unit(
    frame: pd.DataFrame,
    unit: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    point = fixed_binary_metrics(
        frame["y_true"].to_numpy(),
        frame["y_pred"].to_numpy(),
    )["balanced_accuracy"]
    rng = np.random.default_rng(seed)
    if unit in {"window", "participant_task"}:
        cm = confusion_matrix(
            frame["y_true"],
            frame["y_pred"],
            labels=[0, 1],
        ).astype(float)
        probabilities = cm.reshape(-1) / cm.sum()
        sampled = rng.multinomial(
            len(frame),
            probabilities,
            size=n_boot,
        ).reshape(n_boot, 2, 2)
        values = ba_from_confusions(sampled)
    elif unit == "participant":
        matrices = np.stack(
            [
                confusion_matrix(
                    group["y_true"],
                    group["y_pred"],
                    labels=[0, 1],
                ).astype(float)
                for _, group in frame.groupby("subject_id", sort=True)
            ],
            axis=0,
        )
        selected = rng.integers(
            0,
            len(matrices),
            size=(n_boot, len(matrices)),
        )
        counts = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=len(matrices)),
            1,
            selected,
        )
        sampled = np.einsum(
            "bs,sij->bij",
            counts,
            matrices,
            optimize=True,
        )
        values = ba_from_confusions(sampled)
    else:
        raise ValueError(unit)
    return (
        point,
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def paired_bootstrap_by_unit(
    frame: pd.DataFrame,
    unit: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    def difference(data: pd.DataFrame) -> float:
        left = fixed_binary_metrics(
            data["left_true"],
            data["left_pred"],
        )["balanced_accuracy"]
        right = fixed_binary_metrics(
            data["right_true"],
            data["right_pred"],
        )["balanced_accuracy"]
        return left - right

    point = difference(frame)
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=float)
    if unit in {"window", "participant_task"}:
        joint = (
            frame["left_true"].astype(int) * 8
            + frame["left_pred"].astype(int) * 4
            + frame["right_true"].astype(int) * 2
            + frame["right_pred"].astype(int)
        )
        counts = np.bincount(joint, minlength=16)
        samples = rng.multinomial(
            len(frame),
            counts / counts.sum(),
            size=n_boot,
        )
        state = np.arange(16)
        left_true = (state // 8) % 2
        left_pred = (state // 4) % 2
        right_true = (state // 2) % 2
        right_pred = state % 2
        for index, sample_counts in enumerate(samples):
            left_cm = np.zeros((2, 2), dtype=float)
            right_cm = np.zeros((2, 2), dtype=float)
            for state_index, count in enumerate(sample_counts):
                left_cm[
                    left_true[state_index],
                    left_pred[state_index],
                ] += count
                right_cm[
                    right_true[state_index],
                    right_pred[state_index],
                ] += count
            values[index] = (
                ba_from_confusions(left_cm[None, :, :])[0]
                - ba_from_confusions(right_cm[None, :, :])[0]
            )
    elif unit == "participant":
        groups = [
            group
            for _, group in frame.groupby("subject_id", sort=True)
        ]
        for index in range(n_boot):
            selected = rng.integers(0, len(groups), size=len(groups))
            sample = pd.concat(
                [groups[item] for item in selected],
                ignore_index=True,
            )
            values[index] = difference(sample)
    else:
        raise ValueError(unit)
    return (
        point,
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def aggregate_retained_events(windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model", "target", "subject_id", "task_id"]
    for key, group in windows.groupby(keys, sort=True):
        truth = group["y_true"].unique()
        if len(truth) != 1:
            raise RuntimeError(f"retained truth varies in event {key}")
        score = float(group["y_score"].mean())
        rows.append(
            {
                **dict(zip(keys, key)),
                "event_id": f"{key[2]}|{key[3]}",
                "y_true": int(truth[0]),
                "y_score": score,
                "y_pred": int(score >= 0.5),
                "n_windows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def run_uncertainty_by_unit(
    retained_path: Path,
    models: list[str],
    n_boot: int,
    logger: RunLogger,
) -> pd.DataFrame:
    retained = pd.read_parquet(retained_path)
    windows = retained[
        (retained["prediction_level"] == "window")
        & retained["model"].isin(models)
        & retained["target"].isin(
            ["task_condition", "stress_binary_dataset"]
        )
    ].copy()
    windows["subject_id"] = windows["subject_id"].astype(str)
    windows["task_id"] = windows["task_id"].astype(str)
    windows["y_true"] = windows["y_true"].astype(int)
    windows["y_pred"] = windows["y_pred"].astype(int)
    events = aggregate_retained_events(windows)
    rows = []
    for model in models:
        model_windows = windows[windows["model"] == model]
        model_events = events[events["model"] == model]
        for target in ["task_condition", "stress_binary_dataset"]:
            window_target = model_windows[
                model_windows["target"] == target
            ].copy()
            event_target = model_events[
                model_events["target"] == target
            ].copy()
            for unit, frame in [
                ("window", window_target),
                ("participant_task", event_target),
                ("participant", event_target),
            ]:
                point, low, high = bootstrap_binary_by_unit(
                    frame,
                    unit,
                    n_boot,
                    stable_seed("uncertainty", model, target, unit),
                )
                rows.append(
                    {
                        "analysis_class": "diagnostic_only",
                        "outcome": target,
                        "model": model,
                        "resampling_unit": unit,
                        "point_estimate": point,
                        "ci_low": low,
                        "ci_high": high,
                        "ci_width": high - low,
                        "n_bootstrap": n_boot,
                        "n_windows": int(len(window_target)),
                        "n_participant_task_events": int(
                            len(event_target)
                        ),
                        "n_participants": int(
                            event_target["subject_id"].nunique()
                        ),
                        "statistic_level": (
                            "window"
                            if unit == "window"
                            else "participant_task"
                        ),
                        "inference_status": (
                            "invalid_for_primary_inference"
                            if unit == "window"
                            else (
                                "primary_cluster_inference"
                                if unit == "participant"
                                else "event_cluster_diagnostic"
                            )
                        ),
                        "metric": "balanced_accuracy",
                    }
                )

        left_window = model_windows[
            model_windows["target"] == "task_condition"
        ][
            [
                "window_id",
                "subject_id",
                "task_id",
                "y_true",
                "y_pred",
            ]
        ].rename(
            columns={"y_true": "left_true", "y_pred": "left_pred"}
        )
        right_window = model_windows[
            model_windows["target"] == "stress_binary_dataset"
        ][
            [
                "window_id",
                "subject_id",
                "task_id",
                "y_true",
                "y_pred",
            ]
        ].rename(
            columns={"y_true": "right_true", "y_pred": "right_pred"}
        )
        paired_windows = left_window.merge(
            right_window,
            on=["window_id", "subject_id", "task_id"],
            how="inner",
            validate="one_to_one",
        )
        left_event = model_events[
            model_events["target"] == "task_condition"
        ][
            [
                "subject_id",
                "task_id",
                "y_true",
                "y_pred",
            ]
        ].rename(
            columns={"y_true": "left_true", "y_pred": "left_pred"}
        )
        right_event = model_events[
            model_events["target"] == "stress_binary_dataset"
        ][
            [
                "subject_id",
                "task_id",
                "y_true",
                "y_pred",
            ]
        ].rename(
            columns={"y_true": "right_true", "y_pred": "right_pred"}
        )
        paired_events = left_event.merge(
            right_event,
            on=["subject_id", "task_id"],
            how="inner",
            validate="one_to_one",
        )
        for unit, frame in [
            ("window", paired_windows),
            ("participant_task", paired_events),
            ("participant", paired_events),
        ]:
            point, low, high = paired_bootstrap_by_unit(
                frame,
                unit,
                n_boot,
                stable_seed(
                    "uncertainty",
                    model,
                    "task_minus_stress",
                    unit,
                ),
            )
            rows.append(
                {
                    "analysis_class": "diagnostic_only",
                    "outcome": (
                        "task_condition_minus_stress_binary_dataset"
                    ),
                    "model": model,
                    "resampling_unit": unit,
                    "point_estimate": point,
                    "ci_low": low,
                    "ci_high": high,
                    "ci_width": high - low,
                    "n_bootstrap": n_boot,
                    "n_windows": int(len(paired_windows)),
                    "n_participant_task_events": int(
                        len(paired_events)
                    ),
                    "n_participants": int(
                        paired_events["subject_id"].nunique()
                    ),
                    "statistic_level": (
                        "window"
                        if unit == "window"
                        else "participant_task"
                    ),
                    "inference_status": (
                        "invalid_for_primary_inference"
                        if unit == "window"
                        else (
                            "primary_cluster_inference"
                            if unit == "participant"
                            else "event_cluster_diagnostic"
                        )
                    ),
                    "metric": (
                        "balanced_accuracy_left_minus_right"
                    ),
                }
            )
        logger(f"uncertainty comparison completed for {model}")
    return pd.DataFrame(rows)


def summarize_binary_with_subject_ci(
    events: pd.DataFrame,
    model: str,
    design: str,
    n_boot: int,
) -> list[dict[str, object]]:
    rows = []
    for metric in ["balanced_accuracy", "macro_f1"]:
        estimate = metric_value(events, metric)
        low, high, valid = subject_bootstrap_ci(
            events,
            metric,
            n_boot,
            stable_seed("subject_task_disjoint", model, design, metric),
        )
        rows.append(
            {
                "analysis_class": "post_hoc_explanatory",
                "row_type": "performance",
                "model": model,
                "design": design,
                "metric": metric,
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "n_bootstrap": n_boot,
                "valid_bootstrap_resamples": valid,
                "bootstrap_unit": "participant",
                "n_subjects": int(events["subject_id"].nunique()),
                "n_events": int(len(events)),
                "n_windows": int(events["n_windows"].sum()),
                "evaluation_unit": "participant_task",
            }
        )
    return rows


def run_subject_task_disjoint(
    features: pd.DataFrame,
    retained_path: Path,
    n_boot: int,
    logger: RunLogger,
    max_projected_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = features[
        features["task_condition"].notna()
        & ~features["task_id"].astype(str).isin(VIDEO_TASKS)
    ].copy()
    windows["subject_id"] = windows["subject_id"].astype(str)
    windows["task_id"] = windows["task_id"].astype(str)
    windows["event_id"] = (
        windows["subject_id"] + "|" + windows["task_id"]
    )
    windows["y_true"] = windows["task_condition"].astype(int)
    event_keys = (
        windows[["subject_id", "task_id", "event_id"]]
        .drop_duplicates()
        .sort_values(["subject_id", "task_id"])
        .reset_index(drop=True)
    )
    raw = windows[CORE_FEATURE_COLS].to_numpy(dtype=float)
    subjects = windows["subject_id"].to_numpy(dtype=str)
    tasks = windows["task_id"].to_numpy(dtype=str)
    probe_times = []
    for probe in event_keys.iloc[: min(3, len(event_keys))].to_dict(
        "records"
    ):
        train_idx = np.flatnonzero(
            (subjects != probe["subject_id"])
            & (tasks != probe["task_id"])
        )
        test_idx = np.flatnonzero(
            (subjects == probe["subject_id"])
            & (tasks == probe["task_id"])
        )
        start = time.perf_counter()
        x_train, x_test, _, _, _ = robust_impute_scale(
            raw[train_idx],
            raw[test_idx],
        )
        weights = event_equal_weights(
            windows.iloc[train_idx],
            ["subject_id", "task_id"],
        )
        model = make_model("RandomForest")
        model.fit(
            x_train,
            windows.iloc[train_idx]["y_true"].to_numpy(dtype=int),
            sample_weight=weights,
        )
        model.predict_proba(x_test)
        probe_times.append(time.perf_counter() - start)
    projected = float(np.median(probe_times) * len(event_keys))
    logger(
        f"subject-task-disjoint projected runtime: "
        f"{projected:.1f}s from {len(probe_times)} probes"
    )
    if projected > max_projected_seconds:
        feasibility = pd.DataFrame(
            [
                {
                    "analysis_class": "post_hoc_explanatory",
                    "row_type": "feasibility",
                    "model": "RandomForest",
                    "design": "subject_and_task_disjoint",
                    "metric": "",
                    "estimate": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_bootstrap": n_boot,
                    "n_subjects": int(
                        event_keys["subject_id"].nunique()
                    ),
                    "n_events": int(len(event_keys)),
                    "run_status": "not_run_runtime_threshold",
                    "projected_runtime_seconds": projected,
                    "runtime_threshold_seconds": (
                        max_projected_seconds
                    ),
                }
            ]
        )
        return pd.DataFrame(), feasibility

    prediction_rows = []
    for event_number, event in enumerate(
        event_keys.to_dict("records"),
        start=1,
    ):
        heldout_subject = str(event["subject_id"])
        heldout_task = str(event["task_id"])
        train_idx = np.flatnonzero(
            (subjects != heldout_subject) & (tasks != heldout_task)
        )
        test_idx = np.flatnonzero(
            (subjects == heldout_subject) & (tasks == heldout_task)
        )
        if set(subjects[train_idx]) & {heldout_subject}:
            raise RuntimeError("subject exclusion failed")
        if set(tasks[train_idx]) & {heldout_task}:
            raise RuntimeError("task exclusion failed")
        try:
            x_train, x_test, _, _, _ = robust_impute_scale(
                raw[train_idx],
                raw[test_idx],
            )
            weights = event_equal_weights(
                windows.iloc[train_idx],
                ["subject_id", "task_id"],
            )
            model = make_model("RandomForest")
            model.fit(
                x_train,
                windows.iloc[train_idx]["y_true"].to_numpy(dtype=int),
                sample_weight=weights,
            )
            score = float(model.predict_proba(x_test)[:, 1].mean())
            truth = windows.iloc[test_idx]["y_true"].unique()
            if len(truth) != 1:
                raise RuntimeError("held-out event truth is not constant")
            prediction_rows.append(
                {
                    "subject_id": heldout_subject,
                    "task_id": heldout_task,
                    "event_id": str(event["event_id"]),
                    "y_true": int(truth[0]),
                    "y_score": score,
                    "y_pred": int(score >= 0.5),
                    "n_windows": int(len(test_idx)),
                    "train_subject_excluded": True,
                    "train_task_excluded": True,
                }
            )
        except Exception as exc:
            logger.exception(
                f"subject-task-disjoint {heldout_subject}|{heldout_task}",
                exc,
            )
            raise
        if event_number == 1 or event_number % 25 == 0:
            logger(
                f"subject-task-disjoint event "
                f"{event_number}/{len(event_keys)}"
            )
    disjoint = pd.DataFrame(prediction_rows)

    retained = pd.read_parquet(retained_path)
    standard_windows = retained[
        (retained["prediction_level"] == "window")
        & (retained["target"] == "task_condition")
        & (retained["model"] == "RandomForest")
    ].copy()
    standard = aggregate_retained_events(standard_windows)
    standard = standard[
        standard["event_id"].isin(disjoint["event_id"])
    ].copy()
    result_rows = []
    result_rows.extend(
        summarize_binary_with_subject_ci(
            standard,
            "RandomForest",
            "standard_strict_LOSO",
            n_boot,
        )
    )
    result_rows.extend(
        summarize_binary_with_subject_ci(
            disjoint,
            "RandomForest",
            "subject_and_task_disjoint",
            n_boot,
        )
    )
    result = pd.DataFrame(result_rows)
    result["run_status"] = "completed"
    result["projected_runtime_seconds"] = projected
    result["runtime_threshold_seconds"] = max_projected_seconds
    return disjoint, result


def add_context_figure(
    results: pd.DataFrame,
    figure_dir: Path,
) -> None:
    performance = results[results["row_type"] == "performance"].copy()
    panels = [
        ("stress", "spearman_rho", "Stress rating", "Spearman rho"),
        ("arousal", "spearman_rho", "Arousal rating", "Spearman rho"),
        (
            BINARY_TARGET,
            "balanced_accuracy",
            "Dataset binary stress",
            "Balanced accuracy",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2))
    colors = {
        "RandomForest": "#2A6F97",
        "XGBoost": "#D1495B",
        "LightGBM": "#3A7D44",
    }
    x_labels = ["Context", "Physiology", "Combined"]
    x = np.arange(3)
    for axis, (target, metric, title, ylabel) in zip(axes, panels):
        subset = performance[
            (performance["target"] == target)
            & (performance["metric"] == metric)
        ]
        context = subset[
            subset["predictor_set"] == "context_only"
        ].iloc[0]
        for offset, model in zip([-0.12, 0.0, 0.12], MODELS):
            phys = subset[
                (subset["model"] == model)
                & (subset["predictor_set"] == "physiology_only")
            ].iloc[0]
            combined = subset[
                (subset["model"] == model)
                & (
                    subset["predictor_set"]
                    == "context_plus_physiology"
                )
            ].iloc[0]
            selected = pd.DataFrame([context, phys, combined])
            values = selected["estimate"].to_numpy(dtype=float)
            positions = x + offset
            axis.vlines(
                positions,
                selected["ci_low"].to_numpy(dtype=float),
                selected["ci_high"].to_numpy(dtype=float),
                color=colors[model],
                linewidth=1.6,
            )
            axis.scatter(
                positions,
                values,
                color=colors[model],
                marker="o",
                s=24,
                label=model,
                zorder=3,
            )
        axis.set_title(title)
        axis.set_xticks(x, x_labels)
        axis.set_ylabel(ylabel)
        reference = 0.5 if metric == "balanced_accuracy" else 0.0
        axis.axhline(reference, color="#777777", linewidth=0.7)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.6)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(
        "StressID participant-task context and physiology comparison",
        fontsize=12,
    )
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(
            figure_dir / f"context_incremental_value.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def add_residual_figure(
    residual_results: pd.DataFrame,
    figure_dir: Path,
) -> None:
    data = residual_results[
        (residual_results["analysis_scope"] == "overall")
        & (residual_results["metric"] == "rmse")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    order = [
        "ZeroResidualBaseline",
        "RandomForest",
        "XGBoost",
        "LightGBM",
    ]
    colors = ["#7A7A7A", "#2A6F97", "#D1495B", "#3A7D44"]
    for axis, target in zip(axes, CONTINUOUS_TARGETS):
        selected = (
            data[data["target"] == target]
            .set_index("model")
            .loc[order]
        )
        values = selected["estimate"].to_numpy(dtype=float)
        axis.bar(
            np.arange(len(order)),
            values,
            color=colors,
            width=0.72,
        )
        axis.vlines(
            np.arange(len(order)),
            selected["ci_low"].to_numpy(dtype=float),
            selected["ci_high"].to_numpy(dtype=float),
            color="#222222",
            linewidth=1.2,
        )
        axis.set_title(f"{target.capitalize()} residual")
        axis.set_xticks(
            np.arange(len(order)),
            ["Zero", "RF", "XGB", "LGBM"],
        )
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.6)
    axes[0].set_ylabel("RMSE")
    fig.suptitle(
        "Within-task residual prediction under strict LOSO",
        fontsize=12,
    )
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(
            figure_dir / f"residual_prediction.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def add_uncertainty_figure(
    uncertainty: pd.DataFrame,
    figure_dir: Path,
) -> None:
    outcomes = [
        "task_condition",
        "stress_binary_dataset",
        "task_condition_minus_stress_binary_dataset",
    ]
    titles = [
        "Task condition BA",
        "Binary stress BA",
        "Task minus stress BA",
    ]
    units = ["window", "participant_task", "participant"]
    unit_labels = ["Naive window", "Event cluster", "Participant cluster"]
    colors = {
        "RandomForest": "#2A6F97",
        "XGBoost": "#D1495B",
        "LightGBM": "#3A7D44",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), sharey=False)
    width = 0.24
    for axis, outcome, title in zip(axes, outcomes, titles):
        subset = uncertainty[uncertainty["outcome"] == outcome]
        for model_index, model in enumerate(MODELS):
            selected = (
                subset[subset["model"] == model]
                .set_index("resampling_unit")
                .loc[units]
            )
            axis.bar(
                np.arange(3) + (model_index - 1) * width,
                selected["ci_width"].to_numpy(dtype=float),
                width=width,
                color=colors[model],
                label=model,
            )
        axis.set_title(title)
        axis.set_xticks(np.arange(3), unit_labels, rotation=20, ha="right")
        axis.set_ylabel("95% CI width")
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.6)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(
        "Uncertainty changes with the resampling unit",
        fontsize=12,
    )
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(
            figure_dir / f"uncertainty_by_unit.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "(no rows)"
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: (
                ""
                if pd.isna(value)
                else f"{float(value):.{digits}f}"
            )
        )
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(str(value).replace("|", "/") for value in row)
            + " |"
        )
    return "\n".join(lines)


def build_numbers(
    context_results: pd.DataFrame,
    residual_results: pd.DataFrame,
    variance: pd.DataFrame,
    uncertainty: pd.DataFrame,
    disjoint: pd.DataFrame,
    plan_hash: str,
) -> dict[str, object]:
    numbers: dict[str, object] = {
        "analysis_plan_sha256": plan_hash,
        "bootstrap_resamples": int(
            context_results["n_bootstrap"].max()
        ),
        "primary_unit": "participant_task",
        "outer_validation": "strict_LOSO",
    }
    for row in context_results.to_dict("records"):
        if row["row_type"] == "performance":
            key = "_".join(
                [
                    "stressid",
                    "context",
                    slug(row["target"]),
                    slug(row["model"]),
                    slug(row["predictor_set"]),
                    slug(row["metric"]),
                ]
            )
        else:
            key = "_".join(
                [
                    "stressid",
                    "context_difference",
                    slug(row["target"]),
                    slug(row["model"]),
                    slug(row["comparison"]),
                    slug(row["metric"]),
                ]
            )
        numbers[key] = row["estimate"]
        numbers[f"{key}_ci_low"] = row["ci_low"]
        numbers[f"{key}_ci_high"] = row["ci_high"]
    overall = residual_results[
        residual_results["analysis_scope"] == "overall"
    ]
    for row in overall.to_dict("records"):
        key = "_".join(
            [
                "stressid",
                "residual",
                slug(row["target"]),
                slug(row["model"]),
                slug(row["metric"]),
            ]
        )
        numbers[key] = row["estimate"]
        numbers[f"{key}_ci_low"] = row["ci_low"]
        numbers[f"{key}_ci_high"] = row["ci_high"]
    for row in variance[
        variance["row_type"] == "mixed_model"
    ].to_dict("records"):
        for column in [
            "participant_random_intercept_variance",
            "residual_variance",
            "marginal_r_squared",
            "conditional_r_squared",
        ]:
            numbers[
                f"stressid_variance_{slug(row['target'])}_{column}"
            ] = row.get(column)
    for row in uncertainty.to_dict("records"):
        key = "_".join(
            [
                "stressid",
                "uncertainty",
                slug(row["outcome"]),
                slug(row["model"]),
                slug(row["resampling_unit"]),
            ]
        )
        numbers[f"{key}_estimate"] = row["point_estimate"]
        numbers[f"{key}_ci_low"] = row["ci_low"]
        numbers[f"{key}_ci_high"] = row["ci_high"]
        numbers[f"{key}_ci_width"] = row["ci_width"]
    for row in disjoint.to_dict("records"):
        if row.get("row_type") == "performance":
            key = "_".join(
                [
                    "stressid",
                    "disjoint",
                    slug(row["design"]),
                    slug(row["metric"]),
                ]
            )
            numbers[key] = row["estimate"]
            numbers[f"{key}_ci_low"] = row["ci_low"]
            numbers[f"{key}_ci_high"] = row["ci_high"]
    return numbers


def build_summary(
    context_results: pd.DataFrame,
    residual_results: pd.DataFrame,
    variance: pd.DataFrame,
    uncertainty: pd.DataFrame,
    disjoint: pd.DataFrame,
    plan_hash: str,
) -> str:
    context_performance = context_results[
        context_results["row_type"] == "performance"
    ][
        [
            "target",
            "model",
            "predictor_set",
            "metric",
            "estimate",
            "ci_low",
            "ci_high",
            "n_subjects",
            "n_events",
        ]
    ]
    differences = context_results[
        context_results["row_type"] == "paired_difference"
    ][
        [
            "target",
            "model",
            "comparison",
            "metric",
            "estimate",
            "ci_low",
            "ci_high",
            "n_subjects",
            "n_events",
        ]
    ]
    residual_overall = residual_results[
        residual_results["analysis_scope"] == "overall"
    ][
        [
            "target",
            "model",
            "metric",
            "estimate",
            "ci_low",
            "ci_high",
            "n_subjects",
            "n_events",
        ]
    ]
    mixed = variance[variance["row_type"] == "mixed_model"][
        [
            "target",
            "n_subjects",
            "n_events",
            "participant_random_intercept_variance",
            "residual_variance",
            "marginal_r_squared",
            "conditional_r_squared",
            "converged",
        ]
    ]
    uncertainty_display = uncertainty[
        [
            "outcome",
            "model",
            "resampling_unit",
            "point_estimate",
            "ci_low",
            "ci_high",
            "ci_width",
            "n_windows",
            "n_participant_task_events",
            "n_participants",
            "inference_status",
        ]
    ]
    lines = [
        "# StressID Context and Incremental Value Addendum",
        "",
        "## Frozen Plan",
        "",
        f"- SHA-256: `{plan_hash}`.",
        "- The plan was frozen before retained outcomes were inspected.",
        "- Primary unit: participant-task; outer validation: strict LOSO.",
        "- Task identity is a diagnostic context baseline and is not treated as a deployment feature.",
        "- The protocol-order baseline uses the fixed task sequence in `stressid_pipeline.py` as one numeric order index. It does not use task one-hot identity.",
        "",
        "## Prespecified Primary: Context, Physiology, and Combined",
        "",
        markdown_table(context_performance),
        "",
        "## Prespecified Primary: Paired Incremental Differences",
        "",
        "Differences are literal left minus right. Positive favors the left model for Spearman rho, balanced accuracy, and macro-F1; negative favors the left model for MAE and RMSE.",
        "",
        markdown_table(differences),
        "",
        "## Prespecified Primary: Within-Task Residual Prediction",
        "",
        markdown_table(residual_overall),
        "",
        "Per-task residual diagnostics, including all requested confidence intervals, are in `residual_prediction_results.csv`.",
        "",
        "## Post-Hoc Explanatory: Variance Decomposition",
        "",
        markdown_table(mixed),
        "",
        "Task identity and protocol order are confounded. These quantities describe task/protocol-context variance and are not causal task effects.",
        "",
        "## Diagnostic Only: Uncertainty by Analysis Unit",
        "",
        markdown_table(uncertainty_display),
        "",
        "The naive window bootstrap is invalid for primary inference because overlapping windows are not independent. Participant-cluster intervals are the primary inference.",
        "",
        "## Post-Hoc Explanatory: Subject-and-Task-Disjoint Transfer",
        "",
        markdown_table(disjoint),
        "",
        "## Analysis Labels",
        "",
        "- Prespecified primary: Section A and overall Section B results.",
        "- Post-hoc explanatory: variance decomposition and subject-and-task-disjoint transfer.",
        "- Diagnostic only: bootstrap-unit comparison and per-task residual breakdown.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare task and physiological predictions at the original rating level."
    )
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=ROOT / "stressid_run" / "stressid_features.csv",
    )
    parser.add_argument(
        "--retained-predictions",
        type=Path,
        default=ROOT / "paper_results_v2" / "stressid_predictions.parquet",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "paper_results_v2",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            ROOT
            / "paper_results_v2"
            / "addendum_context_incremental_value"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=MODELS,
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--run-disjoint",
        action="store_true",
    )
    parser.add_argument(
        "--disjoint-max-projected-seconds",
        type=float,
        default=900.0,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    guard_new_outputs(args.out_dir)
    plan_path = args.out_dir / "analysis_plan_frozen.md"
    if not plan_path.exists():
        raise FileNotFoundError(
            "analysis_plan_frozen.md must exist before running"
        )
    seed_everything(RANDOM_STATE)
    logger = RunLogger(args.out_dir / "run.log")
    logger("starting StressID context incremental value analysis")
    plan_hash = sha256_text(plan_path)
    logger(f"frozen analysis plan sha256={plan_hash}")

    prior_manifest = snapshot_prior_outputs(
        args.source_dir,
        args.out_dir,
    )
    prior_manifest.to_csv(
        args.out_dir / "prior_outputs_manifest.csv",
        index=False,
    )
    features = pd.read_csv(args.feature_csv, low_memory=False)
    required = set(
        [
            "window_id",
            "subject_id",
            "task_id",
            "task_condition",
            BINARY_COLUMN,
            *CONTINUOUS_TARGETS,
            *CORE_FEATURE_COLS,
        ]
    )
    missing = sorted(required - set(features.columns))
    if missing:
        raise RuntimeError(f"missing feature columns: {missing}")
    unknown_tasks = sorted(set(features["task_id"].astype(str)) - set(TASKS))
    if unknown_tasks:
        raise RuntimeError(f"tasks absent from frozen order: {unknown_tasks}")
    if features["window_id"].duplicated().any():
        raise RuntimeError("duplicate window_id in StressID feature matrix")

    event_predictions, context_results, context_diagnostics = (
        run_context_incremental(
            features,
            args.models,
            args.bootstrap,
            logger,
        )
    )
    event_predictions.to_parquet(
        args.out_dir / "context_incremental_event_predictions.parquet",
        index=False,
    )
    context_results.to_csv(
        args.out_dir / "context_incremental_results.csv",
        index=False,
    )
    logger(
        f"saved context results ({len(context_results)} rows) and "
        f"event predictions ({len(event_predictions)} rows)"
    )

    residual_predictions, residual_results, residual_diagnostics = (
        run_residual_prediction(
            features,
            args.models,
            args.bootstrap,
            logger,
        )
    )
    residual_predictions.to_parquet(
        args.out_dir / "residual_event_predictions.parquet",
        index=False,
    )
    residual_results.to_csv(
        args.out_dir / "residual_prediction_results.csv",
        index=False,
    )
    logger(
        f"saved residual results ({len(residual_results)} rows)"
    )

    variance = run_variance_decomposition(features, logger)
    variance.to_csv(
        args.out_dir / "variance_decomposition.csv",
        index=False,
    )

    uncertainty = run_uncertainty_by_unit(
        args.retained_predictions,
        args.models,
        args.bootstrap,
        logger,
    )
    uncertainty.to_csv(
        args.out_dir / "uncertainty_by_unit.csv",
        index=False,
    )

    if args.run_disjoint:
        disjoint_predictions, disjoint_results = (
            run_subject_task_disjoint(
                features,
                args.retained_predictions,
                args.bootstrap,
                logger,
                args.disjoint_max_projected_seconds,
            )
        )
        if not disjoint_predictions.empty:
            disjoint_predictions.to_parquet(
                args.out_dir
                / "subject_task_disjoint_predictions.parquet",
                index=False,
            )
    else:
        disjoint_results = pd.DataFrame(
            [
                {
                    "analysis_class": "post_hoc_explanatory",
                    "row_type": "feasibility",
                    "model": "RandomForest",
                    "design": "subject_and_task_disjoint",
                    "run_status": "not_requested",
                }
            ]
        )
    disjoint_results.to_csv(
        args.out_dir / "subject_task_disjoint_results.csv",
        index=False,
    )

    diagnostics = pd.concat(
        [context_diagnostics, residual_diagnostics],
        ignore_index=True,
        sort=False,
    )
    diagnostics.to_csv(
        args.out_dir / "fold_diagnostics.csv",
        index=False,
    )

    figure_dir = args.out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=False)
    add_context_figure(context_results, figure_dir)
    add_residual_figure(residual_results, figure_dir)
    add_uncertainty_figure(uncertainty, figure_dir)

    numbers = build_numbers(
        context_results,
        residual_results,
        variance,
        uncertainty,
        disjoint_results,
        plan_hash,
    )
    (args.out_dir / "context_incremental_numbers.json").write_text(
        json.dumps(numbers, indent=2, ensure_ascii=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    summary = build_summary(
        context_results,
        residual_results,
        variance,
        uncertainty,
        disjoint_results,
        plan_hash,
    )
    (args.out_dir / "context_incremental_summary.md").write_text(
        summary,
        encoding="utf-8",
    )

    preserved = verify_prior_outputs(
        args.source_dir,
        args.out_dir,
        prior_manifest,
    )
    preserved.to_csv(
        args.out_dir / "prior_outputs_preserved.csv",
        index=False,
    )
    if not bool(preserved["unchanged"].all()):
        changed = preserved.loc[~preserved["unchanged"], "path"].tolist()
        raise RuntimeError(f"prior outputs changed: {changed}")
    logger(
        f"prior output preservation: "
        f"{int(preserved['unchanged'].sum())}/{len(preserved)} unchanged"
    )
    logger("completed StressID context incremental value analysis")


if __name__ == "__main__":
    main()
