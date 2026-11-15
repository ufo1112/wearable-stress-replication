from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import cohen_kappa_score, recall_score


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_common import (  # noqa: E402
    BINARY_LABELS,
    BOOTSTRAP_N,
    RANDOM_STATE,
    RunLogger,
    classification_metrics,
    ensure_no_subject_leakage,
    event_equal_weights,
    robust_impute_scale,
    seed_everything,
    stable_seed,
    subject_cluster_bootstrap_ci,
    summarize_classification,
    summarize_regression,
    write_csv,
)
from stressid_pipeline import CORE_FEATURE_COLS, make_model  # noqa: E402


try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


MODELS = ["RandomForest", "XGBoost", "LightGBM", "1D-CNN"]
TARGETS = {
    "task_condition": {
        "column": "task_condition",
        "exclude_video": True,
        "status": "protocol_defined",
    },
    "stress_binary_dataset": {
        "column": "binary_stress_official",
        "exclude_video": False,
        "status": "dataset_provided_official_paper_definition",
    },
    "arousal_binary_supplementary": {
        "column": "binary_arousal_official",
        "exclude_video": False,
        "status": "dataset_supplementary_definition_not_documented",
    },
    "stress_median_derived": {
        "column": "stress_median",
        "exclude_video": False,
        "status": "analysis_derived_within_subject_median_strict_greater",
    },
    "arousal_median_derived": {
        "column": "arousal_median",
        "exclude_video": False,
        "status": "analysis_derived_within_subject_median_strict_greater",
    },
}
REGRESSION_TARGETS = ["stress", "arousal"]
PRIMARY_PAIRS = [
    ("task_condition", "stress_binary_dataset"),
    ("task_condition", "arousal_binary_supplementary"),
    ("stress_binary_dataset", "arousal_binary_supplementary"),
]
VIDEO_TASKS = {"Video1", "Video2"}
AGGREGATIONS = ["mean_probability", "median_probability", "majority_vote"]


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


class WeightedTinyCNN(nn.Module):
    def __init__(self, channels: int, classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, classes)

    def forward(self, x):
        return self.head(self.net(x.transpose(1, 2)).squeeze(-1))


def load_inputs(feature_dir: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    features_path = feature_dir / "stressid_features.csv"
    sequences_path = feature_dir / "stressid_sequences.npz"
    df = pd.read_csv(features_path, low_memory=False)
    npz = np.load(sequences_path, allow_pickle=True)
    seq_ids = [str(value) for value in npz["window_id"]]
    id_to_index = {window_id: index for index, window_id in enumerate(seq_ids)}
    missing = [window_id for window_id in df["window_id"].astype(str) if window_id not in id_to_index]
    if missing:
        raise RuntimeError(f"{len(missing)} StressID windows have no sequence")
    seq = npz["sequences"][[id_to_index[value] for value in df["window_id"].astype(str)]].astype(np.float32)
    channels = [str(value) for value in npz["channels"]]
    if len(seq) != len(df):
        raise RuntimeError("feature/sequence row mismatch")
    if df["window_id"].duplicated().any():
        raise RuntimeError("duplicate StressID window_id")
    missing_features = [column for column in CORE_FEATURE_COLS if column not in df.columns]
    if missing_features:
        raise RuntimeError(f"missing CORE_FEATURE_COLS: {missing_features}")
    return df, seq, channels


def prepare_target(
    all_features: pd.DataFrame,
    all_sequences: np.ndarray,
    target: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    spec = TARGETS[target]
    mask = all_features[spec["column"]].notna()
    if spec["exclude_video"]:
        mask &= ~all_features["task_id"].astype(str).isin(VIDEO_TASKS)
    positions = np.flatnonzero(mask.to_numpy())
    df = all_features.loc[mask].copy().reset_index(drop=True)
    df["y_true"] = df[spec["column"]].astype(int)
    df["event_id"] = df["subject_id"].astype(str) + "|" + df["task_id"].astype(str)
    for _, event in df.groupby("event_id", sort=False):
        if event["y_true"].nunique() != 1:
            raise RuntimeError(f"{target} is not constant within event {event['event_id'].iloc[0]}")
    return df, all_sequences[positions]


def fit_predict_classical(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    X = df[CORE_FEATURE_COLS].to_numpy(dtype=np.float64)
    X_train, X_test, medians, means, scales = robust_impute_scale(X[train_idx], X[test_idx])
    y_train = df.iloc[train_idx]["y_true"].to_numpy(dtype=int)
    train_frame = df.iloc[train_idx]
    sample_weight = event_equal_weights(train_frame, ["subject_id", "task_id"])
    model = make_model(model_name)
    if model is None:
        raise RuntimeError(f"{model_name} unavailable")
    model.fit(X_train, y_train, sample_weight=sample_weight)
    score = np.asarray(model.predict_proba(X_test)[:, list(model.classes_).index(1)], dtype=float)
    pred = (score >= 0.5).astype(int)
    guard = {
        "imputer_fit_subjects": int(train_frame["subject_id"].nunique()),
        "scaler_fit_subjects": int(train_frame["subject_id"].nunique()),
        "sample_weight_min": float(sample_weight.min()),
        "sample_weight_max": float(sample_weight.max()),
        "imputer_median_hash": array_hash(medians),
        "scaler_mean_hash": array_hash(means),
        "scaler_scale_hash": array_hash(scales),
    }
    return pred, score, guard


def fit_predict_cnn(
    df: pd.DataFrame,
    sequences: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_seed: int,
    epochs: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if torch is None:
        raise RuntimeError("PyTorch unavailable")
    torch.manual_seed(fold_seed)
    torch.set_num_threads(1)
    X_train = sequences[train_idx].astype(np.float32)
    X_test = sequences[test_idx].astype(np.float32)
    y_train = df.iloc[train_idx]["y_true"].to_numpy(dtype=np.int64)
    train_frame = df.iloc[train_idx]
    event_weight = event_equal_weights(train_frame, ["subject_id", "task_id"]).astype(np.float32)
    mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
    std = np.nanstd(X_train, axis=(0, 1), keepdims=True)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    X_train = np.nan_to_num((X_train - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num((X_test - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    counts = np.bincount(
        y_train,
        weights=event_weight.astype(np.float64),
        minlength=2,
    ).astype(np.float32)
    class_weight = np.zeros(2, dtype=np.float32)
    present = counts > 0
    class_weight[present] = counts[present].sum() / (2 * counts[present])
    device = torch.device("cpu")
    model = WeightedTinyCNN(channels=X_train.shape[2], classes=2).to(device)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weight, dtype=torch.float32, device=device),
        reduction="none",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(event_weight, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(fold_seed)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, generator=generator)
    for _ in range(epochs):
        model.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = loss_fn(model(xb), yb)
            loss = (losses * wb).sum() / wb.sum().clamp_min(1e-12)
            loss.backward()
            optimizer.step()
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(X_test), 512):
            xb = torch.tensor(X_test[start : start + 512], dtype=torch.float32, device=device)
            scores.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
    score = np.concatenate(scores).astype(float)
    return (score >= 0.5).astype(int), score, {
        "imputer_fit_subjects": int(train_frame["subject_id"].nunique()),
        "scaler_fit_subjects": int(train_frame["subject_id"].nunique()),
        "sample_weight_min": float(event_weight.min()),
        "sample_weight_max": float(event_weight.max()),
        "sequence_normalization": "train_subjects_only",
        "class_weight_basis": "event_weighted_training_counts",
    }


def run_classifier(
    df: pd.DataFrame,
    sequences: np.ndarray,
    target: str,
    model_name: str,
    logger: RunLogger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = df["subject_id"].astype(str).to_numpy()
    prediction_parts = []
    diagnostics = []
    for fold_number, heldout in enumerate(sorted(np.unique(subjects)), start=1):
        train_idx = np.flatnonzero(subjects != heldout)
        test_idx = np.flatnonzero(subjects == heldout)
        ensure_no_subject_leakage(subjects, train_idx, test_idx)
        try:
            if model_name == "1D-CNN":
                pred, score, guard = fit_predict_cnn(
                    df,
                    sequences,
                    train_idx,
                    test_idx,
                    stable_seed(RANDOM_STATE, target, model_name, heldout),
                )
            else:
                pred, score, guard = fit_predict_classical(df, train_idx, test_idx, model_name)
            fold = df.iloc[test_idx][
                ["window_id", "subject_id", "task_id", "start_sec", "end_sec", "event_id", "y_true"]
            ].copy()
            fold["target"] = target
            fold["model"] = model_name
            fold["fold_subject"] = heldout
            fold["prediction_level"] = "window"
            fold["y_score"] = score
            fold["y_pred"] = pred
            prediction_parts.append(fold)
            y_test = fold["y_true"].to_numpy(dtype=int)
            diagnostics.append(
                {
                    "diagnostic_type": "fold",
                    "target": target,
                    "model": model_name,
                    "fold_subject": heldout,
                    "n_windows": int(len(fold)),
                    "n_events": int(fold["event_id"].nunique()),
                    "target_classes": int(np.unique(y_test).size),
                    "prediction_classes": int(np.unique(pred).size),
                    "single_class_target": bool(np.unique(y_test).size == 1),
                    "single_class_prediction": bool(np.unique(pred).size == 1),
                    "positive_prevalence": float(np.mean(y_test == 1)),
                    **classification_metrics(y_test, pred, BINARY_LABELS),
                    **guard,
                }
            )
            logger(
                f"StressID {target} {model_name} fold {fold_number}/{len(np.unique(subjects))} "
                f"{heldout}: {len(test_idx)} windows"
            )
        except Exception as exc:
            logger.exception(f"StressID {target} {model_name} {heldout}", exc)
            diagnostics.append(
                {
                    "diagnostic_type": "fold_error",
                    "target": target,
                    "model": model_name,
                    "fold_subject": heldout,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame(), pd.DataFrame(diagnostics)


def aggregate_binary_predictions(window_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["target", "model", "subject_id", "task_id", "event_id", "fold_subject"]
    for key, group in window_predictions.groupby(keys, sort=True):
        truth_values = group["y_true"].dropna().astype(int).unique()
        if len(truth_values) != 1:
            raise RuntimeError(f"nonconstant event truth for {key}")
        mean_score = float(group["y_score"].mean())
        median_score = float(group["y_score"].median())
        majority = int(group["y_pred"].mean() >= 0.5)
        scores = {
            "mean_probability": (mean_score, int(mean_score >= 0.5)),
            "median_probability": (median_score, int(median_score >= 0.5)),
            "majority_vote": (float(group["y_pred"].mean()), majority),
        }
        base = dict(zip(keys, key))
        for aggregation, (score, pred) in scores.items():
            rows.append(
                {
                    **base,
                    "prediction_level": "event",
                    "aggregation": aggregation,
                    "y_true": int(truth_values[0]),
                    "y_score": score,
                    "y_pred": pred,
                    "n_windows": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def classification_result_rows(
    window_predictions: pd.DataFrame,
    event_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_rows = []
    event_rows = []
    for (target, model), group in window_predictions.groupby(["target", "model"], sort=True):
        row = {
            "analysis_type": "classification",
            "dataset": "StressID",
            "target": target,
            "model": model,
            "evaluation_unit": "window",
            "aggregation": "none",
            "long_task_control": "train sample_weight=1/windows_in_subject_task",
            **summarize_classification(
                group,
                BINARY_LABELS,
                seed=stable_seed("stressid", target, model, "window"),
            ),
        }
        window_rows.append(row)
    for (target, model, aggregation), group in event_predictions.groupby(
        ["target", "model", "aggregation"], sort=True
    ):
        row = {
            "analysis_type": "classification",
            "dataset": "StressID",
            "target": target,
            "model": model,
            "evaluation_unit": "participant_task",
            "aggregation": aggregation,
            "long_task_control": "train sample_weight=1/windows_in_subject_task",
            **summarize_classification(
                group,
                BINARY_LABELS,
                seed=stable_seed("stressid", target, model, aggregation),
            ),
        }
        event_rows.append(row)
    return pd.DataFrame(window_rows), pd.DataFrame(event_rows)


def run_regression(
    all_features: pd.DataFrame,
    target: str,
    logger: RunLogger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mask = all_features[target].notna()
    df = all_features.loc[mask].copy().reset_index(drop=True)
    df["event_id"] = df["subject_id"].astype(str) + "|" + df["task_id"].astype(str)
    subjects = df["subject_id"].astype(str).to_numpy()
    X = df[CORE_FEATURE_COLS].to_numpy(dtype=np.float64)
    prediction_parts = []
    diagnostics = []
    for fold_number, heldout in enumerate(sorted(np.unique(subjects)), start=1):
        train_idx = np.flatnonzero(subjects != heldout)
        test_idx = np.flatnonzero(subjects == heldout)
        ensure_no_subject_leakage(subjects, train_idx, test_idx)
        try:
            X_train, X_test, medians, means, scales = robust_impute_scale(X[train_idx], X[test_idx])
            y_train = df.iloc[train_idx][target].to_numpy(dtype=float)
            weights = event_equal_weights(df.iloc[train_idx], ["subject_id", "task_id"])
            model = RandomForestRegressor(
                n_estimators=180,
                max_depth=8,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            model.fit(X_train, y_train, sample_weight=weights)
            score = np.asarray(model.predict(X_test), dtype=float)
            fold = df.iloc[test_idx][
                ["window_id", "subject_id", "task_id", "start_sec", "end_sec", "event_id", target]
            ].copy()
            fold = fold.rename(columns={target: "y_true"})
            fold["target"] = target
            fold["model"] = "RandomForestRegressor"
            fold["fold_subject"] = heldout
            fold["prediction_level"] = "window"
            fold["y_score"] = score
            fold["y_pred"] = score
            prediction_parts.append(fold)
            diagnostics.append(
                {
                    "diagnostic_type": "regression_fold",
                    "target": target,
                    "model": "RandomForestRegressor",
                    "fold_subject": heldout,
                    "n_windows": int(len(fold)),
                    "n_events": int(fold["event_id"].nunique()),
                    "imputer_median_hash": array_hash(medians),
                    "scaler_mean_hash": array_hash(means),
                    "scaler_scale_hash": array_hash(scales),
                }
            )
            logger(f"StressID regression {target} fold {fold_number}/{len(np.unique(subjects))} {heldout}")
        except Exception as exc:
            logger.exception(f"StressID regression {target} {heldout}", exc)
    window = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    event_rows = []
    if not window.empty:
        keys = ["target", "model", "subject_id", "task_id", "event_id", "fold_subject"]
        for key, group in window.groupby(keys, sort=True):
            truth = group["y_true"].dropna().unique()
            if len(truth) != 1:
                raise RuntimeError(f"regression target is not event-constant: {key}")
            base = dict(zip(keys, key))
            for aggregation, pred in [
                ("mean_prediction", float(group["y_pred"].mean())),
                ("median_prediction", float(group["y_pred"].median())),
            ]:
                event_rows.append(
                    {
                        **base,
                        "prediction_level": "event",
                        "aggregation": aggregation,
                        "y_true": float(truth[0]),
                        "y_score": pred,
                        "y_pred": pred,
                        "n_windows": int(len(group)),
                    }
                )
    events = pd.DataFrame(event_rows)
    result_rows = []
    for aggregation, group in events.groupby("aggregation", sort=True):
        result_rows.append(
            {
                "analysis_type": "regression",
                "dataset": "StressID",
                "target": target,
                "model": "RandomForestRegressor",
                "evaluation_unit": "participant_task",
                "aggregation": aggregation,
                "long_task_control": "train sample_weight=1/windows_in_subject_task",
                **summarize_regression(
                    group,
                    seed=stable_seed("stressid", target, "regression", aggregation),
                ),
            }
        )
    return window, events, pd.concat([pd.DataFrame(result_rows), pd.DataFrame(diagnostics)], ignore_index=True, sort=False)


def paired_differences(event_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = event_predictions[event_predictions["aggregation"].isin(AGGREGATIONS)]
    for model in sorted(primary["model"].unique()):
        model_df = primary[primary["model"] == model]
        for aggregation in AGGREGATIONS:
            agg_df = model_df[model_df["aggregation"] == aggregation]
            for left, right in PRIMARY_PAIRS:
                left_df = agg_df[agg_df["target"] == left][
                    ["subject_id", "task_id", "y_true", "y_pred"]
                ].rename(columns={"y_true": "left_true", "y_pred": "left_pred"})
                right_df = agg_df[agg_df["target"] == right][
                    ["subject_id", "task_id", "y_true", "y_pred"]
                ].rename(columns={"y_true": "right_true", "y_pred": "right_pred"})
                merged = left_df.merge(right_df, on=["subject_id", "task_id"], how="inner")
                if merged.empty:
                    continue

                def difference(frame: pd.DataFrame) -> float:
                    left_ba = recall_score(
                        frame["left_true"],
                        frame["left_pred"],
                        labels=BINARY_LABELS,
                        average="macro",
                        zero_division=0,
                    )
                    right_ba = recall_score(
                        frame["right_true"],
                        frame["right_pred"],
                        labels=BINARY_LABELS,
                        average="macro",
                        zero_division=0,
                    )
                    return float(left_ba - right_ba)

                observed = difference(merged)
                subject_pairs = []
                for _, subject_frame in merged.groupby("subject_id", sort=True):
                    left_matrix = np.zeros((2, 2), dtype=float)
                    right_matrix = np.zeros((2, 2), dtype=float)
                    for truth, pred in zip(subject_frame["left_true"], subject_frame["left_pred"]):
                        left_matrix[int(truth), int(pred)] += 1
                    for truth, pred in zip(subject_frame["right_true"], subject_frame["right_pred"]):
                        right_matrix[int(truth), int(pred)] += 1
                    subject_pairs.append((left_matrix, right_matrix))
                left_matrices = np.stack([pair[0] for pair in subject_pairs])
                right_matrices = np.stack([pair[1] for pair in subject_pairs])
                rng = np.random.default_rng(
                    stable_seed("stressid", "paired", model, aggregation, left, right)
                )
                n_subjects = len(subject_pairs)
                selected = rng.integers(0, n_subjects, size=(2000, n_subjects))
                counts = np.apply_along_axis(
                    lambda sample: np.bincount(sample, minlength=n_subjects),
                    axis=1,
                    arr=selected,
                )
                left_boot = np.einsum("bs,sij->bij", counts, left_matrices, optimize=True)
                right_boot = np.einsum("bs,sij->bij", counts, right_matrices, optimize=True)

                def ba_from_cm(cms: np.ndarray) -> np.ndarray:
                    diagonal = np.diagonal(cms, axis1=-2, axis2=-1)
                    support = cms.sum(axis=-1)
                    recall = np.divide(
                        diagonal,
                        support,
                        out=np.zeros_like(diagonal),
                        where=support > 0,
                    )
                    return recall.mean(axis=-1)

                boot_difference = ba_from_cm(left_boot) - ba_from_cm(right_boot)
                low, high = (
                    float(np.percentile(boot_difference, 2.5)),
                    float(np.percentile(boot_difference, 97.5)),
                )
                rows.append(
                    {
                        "dataset": "StressID",
                        "model": model,
                        "aggregation": aggregation,
                        "left_target": left,
                        "right_target": right,
                        "left_minus_right_balanced_accuracy": observed,
                        "ci_low": low,
                        "ci_high": high,
                        "n_subjects": int(merged["subject_id"].nunique()),
                        "n_paired_events": int(len(merged)),
                    }
                )
    return pd.DataFrame(rows)


def permutation_null(event_predictions: pd.DataFrame, n_perm: int = 2000) -> pd.DataFrame:
    rows = []
    subset = event_predictions[
        (event_predictions["model"] == "RandomForest")
        & (event_predictions["aggregation"] == "mean_probability")
        & event_predictions["target"].isin(
            ["task_condition", "stress_binary_dataset", "arousal_binary_supplementary"]
        )
    ]
    for target, group in subset.groupby("target", sort=True):
        observed = float(
            recall_score(
                group["y_true"],
                group["y_pred"],
                labels=BINARY_LABELS,
                average="macro",
                zero_division=0,
            )
        )
        rng = np.random.default_rng(stable_seed("stressid", "permutation", target))
        null = np.empty(n_perm, dtype=float)
        base = group.copy().reset_index(drop=True)
        subject_indices = [indices.to_numpy() for _, indices in base.groupby("subject_id").groups.items()]
        for iteration in range(n_perm):
            permuted = base["y_true"].to_numpy(dtype=int).copy()
            for indices in subject_indices:
                permuted[indices] = rng.permutation(permuted[indices])
            null[iteration] = recall_score(
                permuted,
                base["y_pred"],
                labels=BINARY_LABELS,
                average="macro",
                zero_division=0,
            )
        rows.append(
            {
                "dataset": "StressID",
                "target": target,
                "model": "RandomForest",
                "evaluation_unit": "participant_task",
                "aggregation": "mean_probability",
                "null_type": "fixed_oof_prediction_within_subject_label_permutation",
                "n_permutations": n_perm,
                "observed_balanced_accuracy": observed,
                "distance_from_0_5": observed - 0.5,
                "null_mean": float(np.mean(null)),
                "null_p2_5": float(np.percentile(null, 2.5)),
                "null_p97_5": float(np.percentile(null, 97.5)),
                "empirical_p_upper": float((1 + np.sum(null >= observed)) / (n_perm + 1)),
                "n_subjects": int(group["subject_id"].nunique()),
                "n_events": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def label_diagnostics(features: pd.DataFrame) -> pd.DataFrame:
    events = features.drop_duplicates(["subject_id", "task_id"]).copy()
    rows = []
    mappings = [
        ("stress", "stress_median", "binary_stress_official"),
        ("arousal", "arousal_median", "binary_arousal_official"),
    ]
    for raw, derived, dataset_binary in mappings:
        valid = events[[raw, derived, dataset_binary, "subject_id", "task_id"]].dropna()
        tied_subjects = 0
        all_tied_subjects = 0
        for _, group in events.groupby("subject_id", sort=True):
            values = group[raw].dropna().astype(float)
            if values.empty:
                continue
            median = float(values.median())
            if np.any(values == median):
                tied_subjects += 1
            if np.all(values == median):
                all_tied_subjects += 1
        rows.append(
            {
                "diagnostic_type": "label_agreement",
                "target": raw,
                "derived_label": derived,
                "dataset_label": dataset_binary,
                "equal_to_median_assignment": "low (0), because code uses raw > subject_median",
                "subjects_with_any_median_tie": tied_subjects,
                "subjects_all_tied_at_median": all_tied_subjects,
                "cohen_kappa": float(cohen_kappa_score(valid[derived].astype(int), valid[dataset_binary].astype(int)))
                if not valid.empty
                else np.nan,
                "agreement_n_events": int(len(valid)),
            }
        )
    for target, spec in TARGETS.items():
        sub = events[events[spec["column"]].notna()].copy()
        if spec["exclude_video"]:
            sub = sub[~sub["task_id"].isin(VIDEO_TASKS)]
        values = sub[spec["column"]].astype(int)
        rows.append(
            {
                "diagnostic_type": "event_prevalence",
                "target": target,
                "n_subjects": int(sub["subject_id"].nunique()),
                "n_events": int(len(sub)),
                "positive_prevalence": float(values.mean()) if len(values) else np.nan,
                "support_0": int(np.sum(values == 0)),
                "support_1": int(np.sum(values == 1)),
            }
        )
    return pd.DataFrame(rows)


def verify_label_provenance(
    raw_root: Path,
    features: pd.DataFrame,
    out_path: Path,
) -> None:
    events = features.drop_duplicates(["subject_id", "task_id"])
    stress = events.dropna(subset=["stress", "binary_stress_official"])
    arousal = events.dropna(subset=["arousal", "binary_arousal_official"])
    stress_rule = (stress["stress"].astype(float) >= 5).astype(int)
    arousal_rule = (arousal["arousal"].astype(float) >= 6).astype(int)
    stress_mismatch = int(np.sum(stress_rule != stress["binary_stress_official"].astype(int)))
    arousal_mismatch = int(np.sum(arousal_rule != arousal["binary_arousal_official"].astype(int)))
    lines = [
        "# StressID Label Provenance",
        "",
        "## Dataset Files",
        f"- `labels.csv`: `{raw_root / 'labels.csv'}`; dataset-provided `binary-stress` and `affect3-class`.",
        f"- `labels_supplementary.csv`: `{raw_root / 'labels_supplementary.csv'}`; supplementary binary relax/valence/arousal columns.",
        f"- `self_assessments.csv`: `{raw_root / 'self_assessments.csv'}`; task-level 0-10 ratings.",
        "",
        "## Binary Stress",
        "- The official StressID paper states that ratings below 5 are not stressed and ratings equal to or above 5 are stressed.",
        "- Primary source: https://papers.nips.cc/paper/2023/file/5f09bfe6730e9627a9f800d01a8ad5cd-Paper-Datasets_and_Benchmarks.pdf (Section 3.3, page 6 in the PDF).",
        f"- Local empirical check: `stress >= 5` mismatches dataset `binary-stress` in {stress_mismatch}/{len(stress)} participant-task events.",
        "",
        "## Binary Arousal",
        "- The main paper defines only binary stress and the three-class affect label as its two discrete labels; it does not document a binary-arousal threshold.",
        "- The local supplementary CSV contains `binary-arousal`.",
        f"- Local empirical check: `arousal >= 6` mismatches the supplementary value in {arousal_mismatch}/{len(arousal)} participant-task events.",
        "- Because the threshold was inferred from the released values rather than found in the paper, README, technical-information PDF, or local metadata, this packet calls it `arousal_binary_supplementary`, not an official binary arousal label.",
        "",
        "## Derived Labels",
        "- `stress_median` and `arousal_median` are analysis-derived within-subject median splits.",
        "- Values exactly equal to a subject median are assigned low (0), because `stressid_pipeline.add_targets` uses strict `>`.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate held-out StressID predictions for the participant-task analysis.")
    parser.add_argument("--feature-dir", type=Path, default=ROOT / "stressid_run")
    parser.add_argument(
        "--stressid-root", type=Path, required=True,
        help="StressID folder containing the original assessment and label files.",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper_results_v2")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.out_dir / "tmp" / "stressid_prediction_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.out_dir / "run.log")
    seed_everything()
    logger("starting StressID measurement-unit reanalysis")

    features, sequences, channels = load_inputs(args.feature_dir)
    logger(
        f"StressID input: {len(features)} windows, {features.subject_id.nunique()} subjects, "
        f"{features.groupby(['subject_id', 'task_id']).ngroups} participant-task events"
    )
    logger(f"StressID sequence channels: {channels}")
    shared_rating_mask = features["stress"].notna() & features["arousal"].notna()
    shared_feature_matrix = features.loc[shared_rating_mask, CORE_FEATURE_COLS].to_numpy(
        dtype=np.float64
    )
    shared_feature_hash = hashlib.sha256(
        np.ascontiguousarray(shared_feature_matrix).tobytes()
    ).hexdigest()
    logger(
        "stress/arousal feature audit: "
        f"common_windows={int(shared_rating_mask.sum())}, "
        "CORE_FEATURE_COLS mismatch=0 (both targets reference the same source matrix), "
        f"shared_matrix_sha256={shared_feature_hash}, "
        f"stress_only_windows={int((features['stress'].notna() & features['arousal'].isna()).sum())}, "
        f"arousal_only_windows={int((features['stress'].isna() & features['arousal'].notna()).sum())}"
    )
    if "ecg_4hz" in channels:
        logger(
            "CNN AUDIT FLAG: raw ECG waveform is mean-downsampled from 500 Hz to 4 Hz; "
            "CNN is diagnostic-only and unsuitable for the main cardiac-waveform analysis."
        )

    all_prediction_parts = []
    all_diagnostics = [label_diagnostics(features)]
    for target in TARGETS:
        target_df, target_seq = prepare_target(features, sequences, target)
        event_text = (
            target_df[["subject_id", "task_id"]]
            .drop_duplicates()
            .sort_values(["subject_id", "task_id"])
            .to_csv(index=False)
        )
        event_hash = hashlib.sha256(event_text.encode("utf-8")).hexdigest()
        logger(
            f"prepared {target}: {len(target_df)} windows, "
            f"{target_df.event_id.nunique()} events, event_hash={event_hash}"
        )
        for model_name in args.models:
            cache_model = model_name.replace("-", "_")
            if model_name == "1D-CNN":
                cache_model += "__event_weighted_class_v2"
            part_path = parts_dir / f"{target}__{cache_model}.parquet"
            diag_path = parts_dir / f"{target}__{cache_model}__diagnostics.csv"
            try:
                if part_path.exists() and diag_path.exists():
                    pred = pd.read_parquet(part_path)
                    diag = pd.read_csv(diag_path)
                    logger(f"reused completed prediction part {part_path.name}")
                else:
                    pred, diag = run_classifier(target_df, target_seq, target, model_name, logger)
                    pred.to_parquet(part_path, index=False)
                    diag.to_csv(diag_path, index=False)
                all_prediction_parts.append(pred)
                all_diagnostics.append(diag)
            except Exception as exc:
                logger.exception(f"StressID target/model {target}/{model_name}", exc)

    class_window_predictions = (
        pd.concat(all_prediction_parts, ignore_index=True) if all_prediction_parts else pd.DataFrame()
    )
    class_events = aggregate_binary_predictions(class_window_predictions)
    class_window_results, class_event_results = classification_result_rows(
        class_window_predictions, class_events
    )

    regression_window_parts = []
    regression_event_parts = []
    regression_result_parts = []
    for target in REGRESSION_TARGETS:
        try:
            window_cache = parts_dir / f"regression__{target}__window.parquet"
            event_cache = parts_dir / f"regression__{target}__event.parquet"
            result_cache = parts_dir / f"regression__{target}__result.csv"
            diagnostic_cache = parts_dir / f"regression__{target}__diagnostics.csv"
            if all(
                path.exists()
                for path in [window_cache, event_cache, result_cache, diagnostic_cache]
            ):
                window = pd.read_parquet(window_cache)
                event = pd.read_parquet(event_cache)
                result_only = pd.read_csv(result_cache)
                diagnostic_only = pd.read_csv(diagnostic_cache)
                result_and_diag = pd.concat(
                    [result_only, diagnostic_only],
                    ignore_index=True,
                    sort=False,
                )
                event["aggregation"] = event["aggregation"].replace(
                    {
                        "mean_probability": "mean_prediction",
                        "median_probability": "median_prediction",
                    }
                )
                if "aggregation" in result_and_diag:
                    result_and_diag["aggregation"] = result_and_diag["aggregation"].replace(
                        {
                            "mean_probability": "mean_prediction",
                            "median_probability": "median_prediction",
                        }
                    )
                logger(f"reused completed regression part {target}")
            else:
                window, event, result_and_diag = run_regression(features, target, logger)
                result_only = (
                    result_and_diag[result_and_diag["analysis_type"].notna()]
                    if "analysis_type" in result_and_diag
                    else pd.DataFrame()
                )
                diagnostic_only = (
                    result_and_diag[result_and_diag["diagnostic_type"].notna()]
                    if "diagnostic_type" in result_and_diag
                    else pd.DataFrame()
                )
                window.to_parquet(window_cache, index=False)
                event.to_parquet(event_cache, index=False)
                result_only.to_csv(result_cache, index=False)
                diagnostic_only.to_csv(diagnostic_cache, index=False)
            regression_window_parts.append(window)
            regression_event_parts.append(event)
            regression_result_parts.append(
                result_and_diag[result_and_diag["analysis_type"].notna()]
                if "analysis_type" in result_and_diag
                else pd.DataFrame()
            )
            all_diagnostics.append(
                result_and_diag[result_and_diag["diagnostic_type"].notna()]
                if "diagnostic_type" in result_and_diag
                else pd.DataFrame()
            )
        except Exception as exc:
            logger.exception(f"StressID regression {target}", exc)

    regression_window = (
        pd.concat(regression_window_parts, ignore_index=True) if regression_window_parts else pd.DataFrame()
    )
    regression_events = (
        pd.concat(regression_event_parts, ignore_index=True) if regression_event_parts else pd.DataFrame()
    )
    regression_results = (
        pd.concat(regression_result_parts, ignore_index=True) if regression_result_parts else pd.DataFrame()
    )

    all_predictions = pd.concat(
        [class_window_predictions, class_events, regression_window, regression_events],
        ignore_index=True,
        sort=False,
    )
    all_predictions.to_parquet(args.out_dir / "stressid_predictions.parquet", index=False)
    event_results = pd.concat([class_event_results, regression_results], ignore_index=True, sort=False)
    window_results = class_window_results.copy()
    write_csv(event_results, args.out_dir / "stressid_event_results.csv", logger)
    write_csv(window_results, args.out_dir / "stressid_window_results.csv", logger)

    paired = paired_differences(class_events)
    write_csv(paired, args.out_dir / "stressid_paired_differences.csv", logger)
    diagnostics = pd.concat(all_diagnostics, ignore_index=True, sort=False)
    write_csv(diagnostics, args.out_dir / "stressid_fold_diagnostics.csv", logger)
    permutation = permutation_null(class_events, n_perm=args.permutations)
    write_csv(permutation, args.out_dir / "stressid_permutation_null.csv", logger)
    verify_label_provenance(
        args.stressid_root,
        features,
        args.out_dir / "stressid_label_provenance.md",
    )

    summary = class_event_results[
        (class_event_results["model"] == "RandomForest")
        & (class_event_results["aggregation"] == "mean_probability")
    ][
        [
            "target",
            "pooled_balanced_accuracy",
            "balanced_accuracy_ci_low",
            "balanced_accuracy_ci_high",
            "pooled_macro_f1",
            "n_subjects",
            "n_events",
        ]
    ]
    logger("StressID RandomForest primary event results:\n" + summary.to_string(index=False))
    logger("completed StressID measurement-unit reanalysis")


if __name__ == "__main__":
    main()
