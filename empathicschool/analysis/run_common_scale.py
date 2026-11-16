"""Refit and summarize the EmpathicSchool common-scale analyses in Tables S3 and S4."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from models import CHANNELS, FEATURES, MODELS, SEED, fit_predict_cnn, fit_predict_tree, stable_seed


LABELS = [0, 1, 2]
MODALITIES = {"physio_only": "wearable", "multimodal": "multimodal"}
TARGETS = {"observed": "rating_derived", "hybrid": "hybrid"}
VARIANTS = ("A", "B")
PROBABILITIES = ["prob_0", "prob_1", "prob_2"]
GROUP_COLUMNS = ["dataset_variant", "mode", "variant", "model"]
BOOTSTRAP_RESAMPLES = 2000


def fit_group(frame, sequence_data, modality, mode, model):
    frame = frame.copy()
    frame["subject_id"] = frame["subject_id"].astype(str)
    if frame["window_id"].duplicated().any():
        raise ValueError("Prepared data contain duplicate window IDs.")
    if frame.groupby("event_id")[["subject_id", "y_true"]].nunique().ne(1).any().any():
        raise ValueError("A rating interval has inconsistent participant IDs or classes.")
    indices = frame["_sequence_index"].to_numpy(dtype=int)
    if ((indices < 0) | (indices >= len(sequence_data))).any():
        raise ValueError("Prepared window indexes exceed the available sequences.")
    sequences = sequence_data[indices] if model == "1D-CNN" else None
    subjects = frame["subject_id"].to_numpy()
    window_probs = np.empty((len(frame), len(LABELS)), dtype=float)
    for heldout in sorted(set(subjects)):
        train_idx = np.flatnonzero(subjects != heldout)
        test_idx = np.flatnonzero(subjects == heldout)
        if set(frame.iloc[train_idx]["y_true"]) != set(LABELS):
            raise ValueError("A training fold lacks one of the three classes.")
        if model == "1D-CNN":
            seed = stable_seed("empathic", modality, mode, model, heldout)
            scores = fit_predict_cnn(frame, sequences, train_idx, test_idx, seed)
        else:
            scores = fit_predict_tree(frame, FEATURES[modality], train_idx, test_idx, model)
        if (not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()
                or not np.allclose(scores.sum(axis=1), 1, atol=1e-6)):
            raise ValueError("A held-out fold has invalid class probabilities.")
        window_probs[test_idx] = scores

    windows = frame[["subject_id", "event_id", "y_true"]].copy()
    windows[PROBABILITIES] = window_probs
    aggregation = {name: "mean" for name in PROBABILITIES}
    aggregation["y_true"] = "first"
    groups = windows.groupby(["subject_id", "event_id"], sort=True)
    intervals = groups.agg(aggregation)
    intervals["n_windows"] = groups.size()
    intervals["y_pred"] = intervals[PROBABILITIES].to_numpy().argmax(axis=1)
    return intervals.reset_index()


def refit(prepared_dir: Path) -> pd.DataFrame:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    parts = []
    for modality in MODALITIES:
        sequence_path = prepared_dir / f"{modality}_sequences.npz"
        with np.load(sequence_path, allow_pickle=False) as archive:
            sequences = archive["sequences"]
            if list(archive["columns"]) != CHANNELS[modality] or sequences.shape[1:] != (80, len(CHANNELS[modality])):
                raise ValueError("Prepared sequences must have the original channel order and 80 time points.")
        for mode, variant in product(TARGETS, VARIANTS):
            frame = pd.read_parquet(prepared_dir / f"{modality}__{mode}__{variant}.parquet")
            for model in MODELS:
                part = fit_group(frame, sequences, modality, mode, model)
                part = part.assign(dataset_variant=modality, mode=mode, variant=variant, model=model)
                parts.append(part)
                print(f"Completed {modality}/{mode}/{variant}/{model}", flush=True)
    return pd.concat(parts, ignore_index=True)


def metrics(matrices):
    diagonal = np.diagonal(matrices, axis1=-2, axis2=-1)
    actual = matrices.sum(axis=-1)
    predicted = matrices.sum(axis=-2)
    recall = np.divide(diagonal, actual, out=np.zeros_like(diagonal), where=actual > 0)
    total = actual + predicted
    f1 = np.divide(2 * diagonal, total, out=np.zeros_like(diagonal), where=total > 0)
    return {"balanced_accuracy": recall.mean(axis=-1), "macro_f1": f1.mean(axis=-1)}


def subject_matrices(group):
    return np.stack([
        confusion_matrix(rows["y_true"], rows["y_pred"], labels=LABELS)
        for _, rows in group.groupby("subject_id", sort=True)
    ]).astype(float)


def bootstrap(matrices):
    n_subjects = len(matrices)
    selections = np.random.default_rng(SEED).integers(0, n_subjects, size=(BOOTSTRAP_RESAMPLES, n_subjects))
    counts = np.stack([np.bincount(row, minlength=n_subjects) for row in selections])
    draws = np.einsum("bs,sij->bij", counts, matrices, optimize=True)
    return metrics(draws)


def summarize(intervals):
    summary_rows = []
    paired_rows = []
    results = {}
    for (modality, mode, variant, model), group in intervals.groupby(GROUP_COLUMNS, sort=True):
        matrices = subject_matrices(group)
        pooled = matrices.sum(axis=0)
        points = metrics(pooled)
        draws = bootstrap(matrices)
        ba_low, ba_high = np.percentile(draws["balanced_accuracy"], [2.5, 97.5])
        f1_low, f1_high = np.percentile(draws["macro_f1"], [2.5, 97.5])
        summary_rows.append({
            "analysis": variant, "modality": MODALITIES[modality], "target": TARGETS[mode], "model": model,
            "participants": len(matrices), "intervals": len(group), "windows": int(group["n_windows"].sum()),
            **{f"class_{label}_intervals": int(group["y_true"].eq(label).sum()) for label in LABELS},
            "balanced_accuracy": points["balanced_accuracy"], "ba_ci_low": ba_low, "ba_ci_high": ba_high,
            "macro_f1": points["macro_f1"], "f1_ci_low": f1_low, "f1_ci_high": f1_high,
            "confusion_matrix": json.dumps(pooled.astype(int).tolist()), "bootstrap_seed": SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "evaluation_unit": "participant_task_interval",
        })
        results[modality, mode, variant, model] = (group, points, draws)

    for (modality, mode, model), _ in intervals.groupby(["dataset_variant", "mode", "model"], sort=True):
        a, a_points, a_draws = results[modality, mode, "A", model]
        b, b_points, b_draws = results[modality, mode, "B", model]
        if set(a["subject_id"]) != set(b["subject_id"]) or not set(b["event_id"]) <= set(a["event_id"]):
            raise ValueError("The sensitivity analysis must retain the same participants and a subset of intervals.")
        if not np.array_equal(a.set_index("event_id").loc[b["event_id"], "y_true"], b["y_true"]):
            raise ValueError("Class assignments differ between analyses A and B on shared intervals.")
        # The same seed and sorted participants make the bootstrap draws paired.
        for metric in a_points:
            low, high = np.percentile(b_draws[metric] - a_draws[metric], [2.5, 97.5])
            paired_rows.append({
                "modality": MODALITIES[modality], "target": TARGETS[mode], "model": model, "metric": metric,
                "paired_participants": a["subject_id"].nunique(), "A_intervals": len(a), "B_intervals": len(b),
                "common_intervals": len(b), "bootstrap_seed": SEED,
                "estimand": "B minus A on paired participants; each analysis retains its own interval set",
                "A_estimate": a_points[metric], "B_estimate": b_points[metric],
                "difference_B_minus_A": b_points[metric] - a_points[metric], "ci_low": low, "ci_high": high,
                "n_bootstrap": BOOTSTRAP_RESAMPLES,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("--out-dir already exists; choose a new output directory.")

    intervals = refit(args.prepared_dir)
    summary, paired = summarize(intervals)
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out_dir.parent) as temporary:
        staged = Path(temporary) / "results"
        staged.mkdir()
        summary.to_csv(staged / "results_summary.csv", index=False)
        paired.to_csv(staged / "paired_A_vs_B_differences.csv", index=False)
        staged.rename(args.out_dir)


if __name__ == "__main__":
    main()
