from __future__ import annotations

import hashlib
import json
import random
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 123
BOOTSTRAP_N = 2000
BINARY_LABELS = [0, 1]


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(1)
    except Exception:
        pass


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def exception(self, context: str, exc: BaseException) -> None:
        self(f"ERROR {context}: {type(exc).__name__}: {exc}")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
            handle.write("\n")


def write_csv(df: pd.DataFrame, path: Path, logger: RunLogger | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if logger:
        logger(f"wrote {path} ({len(df)} rows)")


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")


def robust_impute_scale(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(X_train, dtype=np.float64)
    test = np.asarray(X_test, dtype=np.float64)
    medians = np.nanmedian(np.where(np.isfinite(train), train, np.nan), axis=0)
    medians[~np.isfinite(medians)] = 0.0
    train = np.where(np.isfinite(train), train, medians)
    test = np.where(np.isfinite(test), test, medians)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled, medians, scaler.mean_.copy(), scaler.scale_.copy()


def event_equal_weights(df: pd.DataFrame, event_cols: list[str]) -> np.ndarray:
    counts = df.groupby(event_cols, sort=False)[event_cols[0]].transform("size").to_numpy(dtype=float)
    weights = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)
    positive = weights > 0
    if positive.any():
        weights = weights / weights[positive].mean()
    return weights


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
) -> dict[str, float | int | str]:
    truth = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(truth, pred, labels=labels)
    row: dict[str, float | int | str] = {
        "balanced_accuracy": float(
            recall_score(truth, pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "ordinal_mae": float(mean_absolute_error(truth, pred)),
        "weighted_kappa": float(cohen_kappa_score(truth, pred, weights="quadratic"))
        if len(np.unique(np.concatenate([truth, pred]))) > 1
        else np.nan,
        "n": int(len(truth)),
        "confusion_matrix": json.dumps(cm.tolist(), separators=(",", ":")),
    }
    if labels == BINARY_LABELS:
        tn, fp, fn, tp = cm.ravel()
        row.update(
            {
                "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
                "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
                "predicted_positive_rate": float(np.mean(pred == 1)),
                "support_0": int(np.sum(truth == 0)),
                "support_1": int(np.sum(truth == 1)),
            }
        )
    return row


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(truth) & np.isfinite(pred)
    truth = truth[mask]
    pred = pred[mask]
    if len(truth) == 0:
        return {"spearman_rho": np.nan, "mae": np.nan, "rmse": np.nan, "n": 0}
    rho = spearmanr(truth, pred).statistic if len(np.unique(truth)) > 1 else np.nan
    return {
        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
        "mae": float(mean_absolute_error(truth, pred)),
        "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
        "n": int(len(truth)),
    }


def subject_cluster_bootstrap_ci(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    n_boot: int = BOOTSTRAP_N,
    seed: int = RANDOM_STATE,
    subject_col: str = "subject_id",
) -> tuple[float, float]:
    groups = {str(subject): group.copy() for subject, group in df.groupby(subject_col, sort=True)}
    subjects = sorted(groups)
    if not subjects:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        selected = rng.choice(subjects, size=len(subjects), replace=True)
        pieces = []
        for copy_index, subject in enumerate(selected):
            piece = groups[str(subject)].copy()
            piece["_bootstrap_subject"] = f"{copy_index}:{subject}"
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        try:
            values[i] = float(metric_fn(sample))
        except Exception:
            values[i] = np.nan
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def subject_metric_summary(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    subject_col: str = "subject_id",
) -> tuple[float, float, int]:
    values = []
    for _, group in df.groupby(subject_col, sort=True):
        try:
            values.append(float(metric_fn(group)))
        except Exception:
            values.append(np.nan)
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if not len(finite):
        return np.nan, np.nan, 0
    return (
        float(np.mean(finite)),
        float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        int(len(finite)),
    )


def summarize_classification(
    events: pd.DataFrame,
    labels: list[int],
    truth_col: str = "y_true",
    pred_col: str = "y_pred",
    n_boot: int = BOOTSTRAP_N,
    seed: int = RANDOM_STATE,
) -> dict[str, float | int | str]:
    pooled = classification_metrics(events[truth_col].to_numpy(), events[pred_col].to_numpy(), labels)
    subject_matrices = []
    for _, group in events.groupby("subject_id", sort=True):
        subject_matrices.append(
            confusion_matrix(
                group[truth_col].to_numpy(dtype=int),
                group[pred_col].to_numpy(dtype=int),
                labels=labels,
            ).astype(np.float64)
        )
    matrices = np.stack(subject_matrices, axis=0)

    def metrics_from_matrices(cms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        diagonal = np.diagonal(cms, axis1=-2, axis2=-1)
        true_support = cms.sum(axis=-1)
        pred_support = cms.sum(axis=-2)
        recalls = np.divide(
            diagonal,
            true_support,
            out=np.zeros_like(diagonal, dtype=float),
            where=true_support > 0,
        )
        denom = 2.0 * diagonal + (pred_support - diagonal) + (true_support - diagonal)
        f1_values = np.divide(
            2.0 * diagonal,
            denom,
            out=np.zeros_like(diagonal, dtype=float),
            where=denom > 0,
        )
        return recalls.mean(axis=-1), f1_values.mean(axis=-1)

    subject_ba, subject_f1 = metrics_from_matrices(matrices)
    n_subjects = len(matrices)
    rng_ba = np.random.default_rng(seed)
    selected_ba = rng_ba.integers(0, n_subjects, size=(n_boot, n_subjects))
    counts_ba = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=n_subjects),
        axis=1,
        arr=selected_ba,
    )
    bootstrap_matrices_ba = np.einsum("bs,sij->bij", counts_ba, matrices, optimize=True)
    bootstrap_ba, _ = metrics_from_matrices(bootstrap_matrices_ba)
    rng_f1 = np.random.default_rng(seed + 1)
    selected_f1 = rng_f1.integers(0, n_subjects, size=(n_boot, n_subjects))
    counts_f1 = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=n_subjects),
        axis=1,
        arr=selected_f1,
    )
    bootstrap_matrices_f1 = np.einsum("bs,sij->bij", counts_f1, matrices, optimize=True)
    _, bootstrap_f1 = metrics_from_matrices(bootstrap_matrices_f1)
    ba_low, ba_high = float(np.percentile(bootstrap_ba, 2.5)), float(np.percentile(bootstrap_ba, 97.5))
    f1_low, f1_high = float(np.percentile(bootstrap_f1, 2.5)), float(np.percentile(bootstrap_f1, 97.5))
    ba_mean = float(np.mean(subject_ba))
    ba_sd = float(np.std(subject_ba, ddof=1)) if n_subjects > 1 else 0.0
    f1_mean = float(np.mean(subject_f1))
    f1_sd = float(np.std(subject_f1, ddof=1)) if n_subjects > 1 else 0.0
    n_ba = n_subjects
    n_f1 = n_subjects
    return {
        **{f"pooled_{key}": value for key, value in pooled.items()},
        "subject_balanced_accuracy_mean": ba_mean,
        "subject_balanced_accuracy_sd": ba_sd,
        "subject_balanced_accuracy_n": n_ba,
        "balanced_accuracy_ci_low": ba_low,
        "balanced_accuracy_ci_high": ba_high,
        "subject_macro_f1_mean": f1_mean,
        "subject_macro_f1_sd": f1_sd,
        "subject_macro_f1_n": n_f1,
        "macro_f1_ci_low": f1_low,
        "macro_f1_ci_high": f1_high,
        "n_subjects": int(events["subject_id"].nunique()),
        "n_events": int(len(events)),
    }


def summarize_regression(
    events: pd.DataFrame,
    truth_col: str = "y_true",
    pred_col: str = "y_pred",
    n_boot: int = BOOTSTRAP_N,
    seed: int = RANDOM_STATE,
) -> dict[str, float | int]:
    pooled = regression_metrics(events[truth_col].to_numpy(), events[pred_col].to_numpy())
    row: dict[str, float | int] = {f"pooled_{key}": value for key, value in pooled.items()}
    groups = []
    subject_metrics = []
    for _, group in events.groupby("subject_id", sort=True):
        truth = group[truth_col].to_numpy(dtype=float)
        pred = group[pred_col].to_numpy(dtype=float)
        groups.append((truth, pred))
        subject_metrics.append(regression_metrics(truth, pred))
    n_subjects = len(groups)
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, n_subjects, size=(n_boot, n_subjects))
    absolute_sum = np.asarray([np.sum(np.abs(truth - pred)) for truth, pred in groups], dtype=float)
    squared_sum = np.asarray([np.sum((truth - pred) ** 2) for truth, pred in groups], dtype=float)
    counts = np.asarray([len(truth) for truth, _ in groups], dtype=float)
    selection_counts = np.apply_along_axis(
        lambda sample: np.bincount(sample, minlength=n_subjects),
        axis=1,
        arr=selected,
    )
    total_n = selection_counts @ counts
    bootstrap_mae = (selection_counts @ absolute_sum) / total_n
    bootstrap_rmse = np.sqrt((selection_counts @ squared_sum) / total_n)
    bootstrap_rho = np.empty(n_boot, dtype=float)
    for index, sample in enumerate(selected):
        truth = np.concatenate([groups[item][0] for item in sample])
        pred = np.concatenate([groups[item][1] for item in sample])
        bootstrap_rho[index] = spearmanr(truth, pred).statistic if len(np.unique(truth)) > 1 else np.nan
    bootstrap_values = {
        "spearman_rho": bootstrap_rho,
        "mae": bootstrap_mae,
        "rmse": bootstrap_rmse,
    }
    for metric_name in ["spearman_rho", "mae", "rmse"]:
        subject_values = np.asarray([metric[metric_name] for metric in subject_metrics], dtype=float)
        finite_subject = subject_values[np.isfinite(subject_values)]
        boot = bootstrap_values[metric_name]
        boot = boot[np.isfinite(boot)]
        row[f"subject_{metric_name}_mean"] = (
            float(np.mean(finite_subject)) if len(finite_subject) else np.nan
        )
        row[f"subject_{metric_name}_sd"] = (
            float(np.std(finite_subject, ddof=1)) if len(finite_subject) > 1 else 0.0
        )
        row[f"subject_{metric_name}_n"] = int(len(finite_subject))
        row[f"{metric_name}_ci_low"] = float(np.percentile(boot, 2.5)) if len(boot) else np.nan
        row[f"{metric_name}_ci_high"] = float(np.percentile(boot, 97.5)) if len(boot) else np.nan
    row["n_subjects"] = int(events["subject_id"].nunique())
    row["n_events"] = int(len(events))
    return row


def ensure_no_subject_leakage(subjects: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    train_subjects = set(np.asarray(subjects)[train_idx].astype(str))
    test_subjects = set(np.asarray(subjects)[test_idx].astype(str))
    overlap = train_subjects & test_subjects
    if overlap:
        raise RuntimeError(f"subject leakage detected: {sorted(overlap)}")


def markdown_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    lines = [
        "| " + " | ".join(str(column) for column in view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in view.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\nShowing first {max_rows} of {len(df)} rows.")
    return "\n".join(lines)


def package_versions(names: Iterable[str]) -> list[dict[str, str]]:
    import importlib.metadata

    rows = []
    for name in names:
        try:
            version = importlib.metadata.version(name)
            rows.append({"package": name, "version": version, "status": "available"})
        except Exception as exc:
            rows.append({"package": name, "version": "", "status": f"{type(exc).__name__}: {exc}"})
    return rows
