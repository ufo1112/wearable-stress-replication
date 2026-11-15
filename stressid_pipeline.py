from __future__ import annotations

import argparse
import hashlib
import math
import random
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    import neurokit2 as nk
except Exception:  # pragma: no cover
    nk = None

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
except Exception:  # pragma: no cover
    RandomOverSampler = None
    SMOTE = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


RANDOM_STATE = 123
BOOTSTRAP_N = 2000
FS_RAW = 500
FS_4HZ = 4
RAW_PER_4HZ = FS_RAW // FS_4HZ
STEP_SEC_DEFAULT = 5
WINDOW_LENGTHS = [20, 10, 30, 60]
TASKS = [
    "Breathing",
    "Video1",
    "Video2",
    "Counting1",
    "Counting2",
    "Counting3",
    "Stroop",
    "Speaking",
    "Math",
    "Reading",
    "Relax",
]
STRESSOR_TASKS = {"Counting1", "Counting2", "Counting3", "Math", "Reading", "Speaking", "Stroop"}
NONSTRESSOR_TASKS = {"Breathing", "Relax", "Video1", "Video2"}
MODELS = ["RandomForest", "XGBoost", "LightGBM", "1D-CNN"]
PRIMARY_TARGETS = ["task_condition", "arousal_median", "stress_median"]
SECONDARY_TARGETS = ["arousal_tertile", "stress_tertile"]
ALL_TARGETS = PRIMARY_TARGETS + SECONDARY_TARGETS
BINARY_LABELS = [0, 1]

CORE_FEATURE_COLS = [
    "eda_tonic_mean",
    "eda_phasic_mean",
    "eda_phasic_std",
    "eda_scr_peak_count",
    "eda_scr_mean_amplitude",
    "eda_scr_max_amplitude",
    "eda_scr_mean_rise_time",
    "eda_phasic_auc",
    "eda_slope",
    "eda_mean",
    "eda_std",
    "eda_min",
    "eda_max",
    "eda_skew",
    "eda_kurtosis",
    "ecg_peak_count",
    "ecg_mean_ibi_ms",
    "ecg_rmssd_ms",
    "ecg_sdnn_ms",
    "ecg_pnn50",
    "ecg_hr_mean_bpm",
    "ecg_mean",
    "ecg_std",
    "ecg_min",
    "ecg_max",
    "ecg_skew",
    "ecg_kurtosis",
]
RESP_FEATURE_COLS = [
    "resp_mean",
    "resp_std",
    "resp_min",
    "resp_max",
    "resp_rate_bpm",
    "resp_peak_count",
    "resp_mean_amplitude",
]
SEQUENCE_CHANNELS = ["ecg_4hz", "eda", "eda_phasic", "eda_tonic"]


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.set_num_threads(1)


def log(message: str, out_dir: Path) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (out_dir / "run.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def fmt(value: object, digits: int = 3) -> str:
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def table_md(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in view.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing first {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def clean_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(arr) == 0:
        return arr
    mask = np.isfinite(arr)
    if mask.all():
        return arr
    if not mask.any():
        return np.zeros_like(arr)
    idx = np.arange(len(arr))
    arr[~mask] = np.interp(idx[~mask], idx[mask], arr[mask])
    return arr


def raw_to_units(raw: pd.DataFrame) -> pd.DataFrame:
    chan_bit = float(2**16)
    vcc = 3.0
    out = pd.DataFrame()
    out["ecg"] = ((raw["ECG"].astype(float) / chan_bit) - 0.5) * vcc
    out["eda"] = ((raw["EDA"].astype(float) / chan_bit) * vcc) / 0.12
    out["resp"] = ((raw["RR"].astype(float) / chan_bit) - 0.5) * 100.0
    return out


def downsample_mean(values: np.ndarray, samples_per_bin: int = RAW_PER_4HZ) -> np.ndarray:
    arr = clean_1d(values)
    usable = (len(arr) // samples_per_bin) * samples_per_bin
    if usable <= 0:
        return np.zeros(0, dtype=np.float64)
    return arr[:usable].reshape(-1, samples_per_bin).mean(axis=1)


def slope(values: np.ndarray, fs: int) -> float:
    y = clean_1d(values)
    if len(y) < 2 or np.nanstd(y) == 0:
        return 0.0
    t = np.arange(len(y), dtype=np.float64) / fs
    try:
        return float(np.polyfit(t, y, 1)[0])
    except Exception:
        return 0.0


def basic_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    y = clean_1d(values)
    if len(y) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_skew": np.nan,
            f"{prefix}_kurtosis": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.nanmean(y)),
        f"{prefix}_std": float(np.nanstd(y)),
        f"{prefix}_min": float(np.nanmin(y)),
        f"{prefix}_max": float(np.nanmax(y)),
        f"{prefix}_skew": float(stats.skew(y, nan_policy="omit")) if len(y) > 2 else 0.0,
        f"{prefix}_kurtosis": float(stats.kurtosis(y, nan_policy="omit")) if len(y) > 3 else 0.0,
    }


def decompose_eda(eda: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    eda = clean_1d(eda)
    if nk is not None and len(eda) >= FS_4HZ * 10:
        try:
            out = nk.eda_phasic(eda, sampling_rate=FS_4HZ, method="neurokit")
            tonic = clean_1d(out["EDA_Tonic"].to_numpy(dtype=np.float64))
            phasic = clean_1d(out["EDA_Phasic"].to_numpy(dtype=np.float64))
            return tonic, phasic, "neurokit2"
        except Exception:
            pass
    try:
        b, a = signal.butter(2, 0.05 / (FS_4HZ / 2.0), btype="highpass")
        phasic = signal.filtfilt(b, a, eda)
        tonic = eda - phasic
        return clean_1d(tonic), clean_1d(phasic), "butter_highpass_0.05hz"
    except Exception:
        return eda.copy(), np.zeros_like(eda), "fallback_zero_phasic"


def eda_features(eda: np.ndarray, tonic: np.ndarray, phasic: np.ndarray) -> dict[str, float]:
    eda = clean_1d(eda)
    tonic = clean_1d(tonic)
    phasic = clean_1d(phasic)
    prom = max(0.005, float(np.nanstd(phasic)) * 0.5)
    peaks, _ = signal.find_peaks(phasic, prominence=prom, distance=FS_4HZ)
    amps = np.array([], dtype=np.float64)
    if len(peaks):
        try:
            amps = signal.peak_prominences(phasic, peaks)[0]
        except Exception:
            amps = phasic[peaks]
    rise_times = []
    for peak in peaks:
        lo = max(0, int(peak) - 4 * FS_4HZ)
        if peak <= lo:
            continue
        trough = lo + int(np.argmin(phasic[lo : peak + 1]))
        rise_times.append((int(peak) - trough) / FS_4HZ)
    out = {
        "eda_tonic_mean": float(np.nanmean(tonic)) if len(tonic) else np.nan,
        "eda_phasic_mean": float(np.nanmean(phasic)) if len(phasic) else np.nan,
        "eda_phasic_std": float(np.nanstd(phasic)) if len(phasic) else np.nan,
        "eda_scr_peak_count": int(len(peaks)),
        "eda_scr_mean_amplitude": float(np.nanmean(amps)) if len(amps) else 0.0,
        "eda_scr_max_amplitude": float(np.nanmax(amps)) if len(amps) else 0.0,
        "eda_scr_mean_rise_time": float(np.nanmean(rise_times)) if rise_times else 0.0,
        "eda_phasic_auc": float(np.trapezoid(np.clip(phasic, 0, None), dx=1.0 / FS_4HZ)) if len(phasic) else 0.0,
        "eda_slope": slope(eda, FS_4HZ),
    }
    out.update(basic_stats("eda", eda))
    return out


def find_ecg_peaks(ecg: np.ndarray, fs: int = FS_RAW) -> np.ndarray:
    x = clean_1d(ecg)
    if len(x) < fs * 5:
        return np.array([], dtype=int)
    try:
        detrended = signal.detrend(x)
        b, a = signal.butter(2, [0.5 / (fs / 2.0), 35.0 / (fs / 2.0)], btype="bandpass")
        filt = signal.filtfilt(b, a, detrended)
    except Exception:
        filt = x - np.nanmean(x)
    distance = int(0.30 * fs)
    prom = max(float(np.nanstd(filt)) * 0.6, 1e-5)
    candidates = []
    for signed in [filt, -filt]:
        peaks, _ = signal.find_peaks(signed, distance=distance, prominence=prom)
        ibi_ms = np.diff(peaks) / fs * 1000.0
        ibi_ok = ibi_ms[(ibi_ms >= 300.0) & (ibi_ms <= 2000.0)]
        score = len(ibi_ok) - 0.15 * abs(len(peaks) - max(len(x) / fs, 1) * 1.2)
        candidates.append((score, peaks))
    return max(candidates, key=lambda item: item[0])[1].astype(int)


def ecg_features(ecg: np.ndarray, fs: int = FS_RAW) -> dict[str, float]:
    y = clean_1d(ecg)
    peaks = find_ecg_peaks(y, fs)
    ibi_ms = np.diff(peaks) / fs * 1000.0
    ibi_ms = ibi_ms[(ibi_ms >= 300.0) & (ibi_ms <= 2000.0)]
    diff_ibi = np.diff(ibi_ms)
    out = {
        "ecg_peak_count": int(len(peaks)),
        "ecg_mean_ibi_ms": float(np.nanmean(ibi_ms)) if len(ibi_ms) else np.nan,
        "ecg_rmssd_ms": float(np.sqrt(np.nanmean(diff_ibi * diff_ibi))) if len(diff_ibi) else 0.0,
        "ecg_sdnn_ms": float(np.nanstd(ibi_ms, ddof=1)) if len(ibi_ms) > 1 else 0.0,
        "ecg_pnn50": float(np.nanmean(np.abs(diff_ibi) > 50.0)) if len(diff_ibi) else 0.0,
        "ecg_hr_mean_bpm": float(60000.0 / np.nanmean(ibi_ms)) if len(ibi_ms) and np.nanmean(ibi_ms) > 0 else np.nan,
    }
    out.update(basic_stats("ecg", y))
    return out


def respiration_features(resp: np.ndarray, fs: int = FS_RAW) -> dict[str, float]:
    y = clean_1d(resp)
    out = {k: v for k, v in basic_stats("resp", y).items() if not k.endswith("_skew") and not k.endswith("_kurtosis")}
    if len(y) < fs * 10:
        out.update({"resp_rate_bpm": np.nan, "resp_peak_count": 0, "resp_mean_amplitude": 0.0})
        return out
    try:
        b, a = signal.butter(2, [0.05 / (fs / 2.0), 0.7 / (fs / 2.0)], btype="bandpass")
        filt = signal.filtfilt(b, a, y)
    except Exception:
        filt = y - np.nanmean(y)
    peaks, _ = signal.find_peaks(filt, distance=int(1.2 * fs), prominence=max(np.nanstd(filt) * 0.25, 1e-6))
    amps = signal.peak_prominences(filt, peaks)[0] if len(peaks) else np.array([], dtype=np.float64)
    minutes = len(y) / fs / 60.0
    out.update(
        {
            "resp_rate_bpm": float(len(peaks) / minutes) if minutes > 0 else np.nan,
            "resp_peak_count": int(len(peaks)),
            "resp_mean_amplitude": float(np.nanmean(amps)) if len(amps) else 0.0,
        }
    )
    return out


def parse_self_assessments(root: Path, out_dir: Path) -> pd.DataFrame:
    raw = pd.read_csv(root / "self_assessments.csv", sep=";")
    subjects = [c for c in raw.columns if c != "Tasks"]
    rows = []
    for _, rec in raw.iterrows():
        task_metric = str(rec["Tasks"])
        if "_" not in task_metric:
            continue
        task, metric = task_metric.rsplit("_", 1)
        if task not in TASKS or metric not in {"relax", "stress", "valence", "arousal"}:
            continue
        for subject in subjects:
            value = pd.to_numeric(pd.Series([rec[subject]]), errors="coerce").iloc[0]
            rows.append({"subject_id": subject, "task_id": task, "metric": metric, "value": value})
    long = pd.DataFrame(rows)
    full_index = pd.MultiIndex.from_product([subjects, TASKS], names=["subject_id", "task_id"])
    wide = (
        long.pivot_table(index=["subject_id", "task_id"], columns="metric", values="value", aggfunc="first")
        .reindex(full_index)
        .reset_index()
    )
    wide.columns.name = None
    for col in ["relax", "stress", "valence", "arousal"]:
        wide[col] = pd.to_numeric(wide[col], errors="coerce")
    labels = pd.read_csv(root / "labels.csv")
    labels[["subject_id", "task_id"]] = labels["subject/task"].str.extract(r"^([^_]+)_(.+)$")
    labels = labels.drop(columns=["subject/task"])
    supp = pd.read_csv(root / "labels_supplementary.csv")
    supp[["subject_id", "task_id"]] = supp["subject/task"].str.extract(r"^([^_]+)_(.+)$")
    supp = supp.drop(columns=["subject/task"])
    wide = wide.merge(labels, on=["subject_id", "task_id"], how="left").merge(supp, on=["subject_id", "task_id"], how="left")
    wide["task_condition"] = np.where(wide["task_id"].isin(STRESSOR_TASKS), 1, 0)
    wide["task_group"] = np.where(wide["task_id"].isin(STRESSOR_TASKS), "stressor", "non_stressor")
    wide.to_csv(out_dir / "stressid_selfreport.csv", index=False)
    return wide


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    unique_tasks = out.drop_duplicates(["subject_id", "task_id"]).copy()
    for metric in ["stress", "arousal"]:
        target = f"{metric}_median"
        out[target] = np.nan
        medians = {}
        for subject, group in unique_tasks.groupby("subject_id"):
            values = group[metric].astype(float)
            if values.notna().sum() == 0:
                continue
            medians[subject] = float(values.median())
        out[target] = out.apply(
            lambda row: np.nan
            if pd.isna(row[metric]) or row["subject_id"] not in medians
            else float(row[metric] > medians[row["subject_id"]]),
            axis=1,
        )
        low = float(unique_tasks[metric].quantile(1.0 / 3.0))
        high = float(unique_tasks[metric].quantile(2.0 / 3.0))
        tert = pd.Series(np.nan, index=out.index, dtype=float)
        tert[out[metric] <= low] = 0.0
        tert[out[metric] >= high] = 1.0
        out[f"{metric}_tertile"] = tert
        out[f"{metric}_dataset_tertile_low_cut"] = low
        out[f"{metric}_dataset_tertile_high_cut"] = high
    return out


def load_task_physio(root: Path, subject: str, task: str) -> pd.DataFrame:
    path = root / "Physiological" / subject / f"{subject}_{task}.txt"
    raw = pd.read_csv(path)
    required = {"ECG", "EDA", "RR"}
    if set(raw.columns) != required:
        missing = required.difference(raw.columns)
        raise RuntimeError(f"{path} unexpected columns={raw.columns.tolist()} missing={missing}")
    return raw_to_units(raw)


def process_task(root: Path, subject: str, task: str, self_row: pd.Series, window_sec: int) -> tuple[list[dict], list[np.ndarray]]:
    data = load_task_physio(root, subject, task)
    ecg = clean_1d(data["ecg"].to_numpy())
    eda = clean_1d(data["eda"].to_numpy())
    resp = clean_1d(data["resp"].to_numpy())
    eda4 = downsample_mean(eda)
    ecg4 = downsample_mean(ecg)
    resp4 = downsample_mean(resp)
    n4 = min(len(eda4), len(ecg4), len(resp4))
    eda4 = eda4[:n4]
    ecg4 = ecg4[:n4]
    resp4 = resp4[:n4]
    tonic, phasic, eda_method = decompose_eda(eda4)
    tonic = tonic[:n4]
    phasic = phasic[:n4]
    win_raw = int(window_sec * FS_RAW)
    step_raw = int(STEP_SEC_DEFAULT * FS_RAW)
    win4 = int(window_sec * FS_4HZ)
    rows: list[dict] = []
    seqs: list[np.ndarray] = []
    max_start = len(ecg) - win_raw
    if max_start < 0:
        return rows, seqs
    for k, start_raw in enumerate(range(0, max_start + 1, step_raw)):
        end_raw = start_raw + win_raw
        start4 = int(round(start_raw / FS_RAW * FS_4HZ))
        end4 = start4 + win4
        if end4 > n4:
            continue
        window_id = f"{subject}_{task}_{window_sec}s_{k:04d}"
        eda_win = eda4[start4:end4]
        tonic_win = tonic[start4:end4]
        phasic_win = phasic[start4:end4]
        ecg_win = ecg[start_raw:end_raw]
        resp_win = resp[start_raw:end_raw]
        row = {
            "window_id": window_id,
            "subject_id": subject,
            "task_id": task,
            "window_sec": int(window_sec),
            "step_sec": STEP_SEC_DEFAULT,
            "start_sec": start_raw / FS_RAW,
            "end_sec": end_raw / FS_RAW,
            "duration_sec": len(ecg) / FS_RAW,
            "fs_ecg_hz": FS_RAW,
            "fs_eda_hz": FS_RAW,
            "fs_resp_hz": FS_RAW,
            "eda_decomposition": eda_method,
            "task_condition": int(self_row["task_condition"]),
            "task_group": str(self_row["task_group"]),
            "stress": float(self_row["stress"]) if pd.notna(self_row["stress"]) else np.nan,
            "relax": float(self_row["relax"]) if pd.notna(self_row["relax"]) else np.nan,
            "arousal": float(self_row["arousal"]) if pd.notna(self_row["arousal"]) else np.nan,
            "valence": float(self_row["valence"]) if pd.notna(self_row["valence"]) else np.nan,
            "binary_stress_official": self_row.get("binary-stress", np.nan),
            "binary_arousal_official": self_row.get("binary-arousal", np.nan),
            "binary_relax_official": self_row.get("binary-relax", np.nan),
            "binary_valence_official": self_row.get("binary-valence", np.nan),
        }
        row.update(eda_features(eda_win, tonic_win, phasic_win))
        row.update(ecg_features(ecg_win))
        row.update(respiration_features(resp_win))
        seq = np.stack([ecg4[start4:end4], eda_win, phasic_win, tonic_win], axis=1).astype(np.float32)
        rows.append(row)
        seqs.append(seq)
    return rows, seqs


def build_features(root: Path, out_dir: Path, self_report: pd.DataFrame, window_sec: int) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict] = []
    seqs: list[np.ndarray] = []
    phys_root = root / "Physiological"
    subjects = sorted([p.name for p in phys_root.iterdir() if p.is_dir() and not p.name.startswith("._")])
    self_lookup = {(str(row["subject_id"]), str(row["task_id"])): row for _, row in self_report.iterrows()}
    failures = []
    for subject in subjects:
        for task in TASKS:
            key = (subject, task)
            path = phys_root / subject / f"{subject}_{task}.txt"
            if key not in self_lookup or not path.exists():
                failures.append({"subject_id": subject, "task_id": task, "reason": "missing_self_or_physio"})
                continue
            try:
                task_rows, task_seqs = process_task(root, subject, task, self_lookup[key], window_sec)
                rows.extend(task_rows)
                seqs.extend(task_seqs)
            except Exception as exc:
                failures.append({"subject_id": subject, "task_id": task, "reason": f"{type(exc).__name__}: {exc}"})
    features = add_targets(pd.DataFrame(rows))
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / f"stressid_feature_failures_{window_sec}s.csv", index=False)
    seq = np.asarray(seqs, dtype=np.float32)
    suffix = "" if window_sec == 20 else f"_{window_sec}s"
    features.to_csv(out_dir / f"stressid_features{suffix}.csv", index=False)
    np.savez_compressed(
        out_dir / f"stressid_sequences{suffix}.npz",
        window_id=features["window_id"].astype(str).to_numpy(),
        sequences=seq,
        channels=np.asarray(SEQUENCE_CHANNELS, dtype=object),
    )
    return features, seq


def robust_impute_scale(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(X_train, dtype=np.float64)
    test = np.asarray(X_test, dtype=np.float64)
    med = np.nanmedian(np.where(np.isfinite(train), train, np.nan), axis=0)
    med[~np.isfinite(med)] = 0.0
    train = np.where(np.isfinite(train), train, med)
    test = np.where(np.isfinite(test), test, med)
    scaler = StandardScaler()
    return scaler.fit_transform(train), scaler.transform(test)


def maybe_smote(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or SMOTE is None or RandomOverSampler is None:
        return X, y
    min_count = int(np.min(counts))
    try:
        if min_count >= 2:
            return SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, min_count - 1)).fit_resample(X, y)
        return RandomOverSampler(random_state=RANDOM_STATE).fit_resample(X, y)
    except Exception:
        return X, y


def make_model(model_name: str):
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=180,
            max_depth=8,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=None,
        )
    if model_name == "XGBoost":
        if XGBClassifier is None:
            return None
        return XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
    if model_name == "LightGBM":
        if LGBMClassifier is None:
            return None
        return LGBMClassifier(
            objective="binary",
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


def fit_predict_classical(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, model_name: str) -> np.ndarray:
    X_train, X_test = robust_impute_scale(X[train_idx], X[test_idx])
    y_train = y[train_idx]
    X_res, y_res = maybe_smote(X_train, y_train)
    model = make_model(model_name)
    if model is None:
        raise RuntimeError(f"{model_name} unavailable")
    model.fit(X_res, y_res)
    return np.asarray(model.predict(X_test), dtype=int)


class TinyCNN(nn.Module):
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


def fit_predict_cnn(seq: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, epochs: int = 4) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch unavailable")
    X_train = seq[train_idx].astype(np.float32)
    X_test = seq[test_idx].astype(np.float32)
    y_train = y[train_idx].astype(np.int64)
    mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
    std = np.nanstd(X_train, axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    X_train = np.nan_to_num((X_train - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num((X_test - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    weights = np.zeros(2, dtype=np.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (2 * counts[present])
    weights[~present] = 0.0
    device = torch.device("cpu")
    model = TinyCNN(channels=X_train.shape[2], classes=2).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=128,
        shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_test), 512):
            xb = torch.tensor(X_test[start : start + 512], dtype=torch.float32, device=device)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds).astype(int)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(recall_score(y_true, y_pred, labels=BINARY_LABELS, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=BINARY_LABELS, average="macro", zero_division=0)),
    }


def append_prediction_rows(
    rows: list[dict],
    dataset: str,
    target: str,
    split: str,
    model: str,
    fold_id: str,
    df: pd.DataFrame,
    idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    for local_i, truth, pred in zip(idx, y_true, y_pred):
        rec = df.iloc[int(local_i)]
        rows.append(
            {
                "dataset": dataset,
                "target": target,
                "split": split,
                "model": model,
                "fold_id": fold_id,
                "subject_id": rec["subject_id"],
                "task_id": rec["task_id"],
                "window_id": rec["window_id"],
                "window_sec": rec["window_sec"],
                "start_sec": rec["start_sec"],
                "y_true": int(truth),
                "y_pred": int(pred),
            }
        )


def evaluate_loso(
    features: pd.DataFrame,
    seq: np.ndarray,
    target: str,
    models: list[str],
    dataset_name: str,
    split_name: str = "strict_loso",
    common_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = features.copy()
    mask = pd.Series(True, index=base.index)
    if common_mask is not None:
        mask &= common_mask.reindex(base.index, fill_value=False)
    mask &= base[target].notna()
    positions = np.flatnonzero(mask.to_numpy())
    df = base.loc[mask].copy().reset_index(drop=True)
    seq = seq[positions]
    if df.empty or df[target].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()
    y = df[target].astype(int).to_numpy()
    X = df[CORE_FEATURE_COLS].to_numpy(dtype=np.float64)
    subjects = df["subject_id"].astype(str).to_numpy()
    pred_rows: list[dict] = []
    fold_rows: list[dict] = []
    for heldout in sorted(np.unique(subjects)):
        train_idx = np.flatnonzero(subjects != heldout)
        test_idx = np.flatnonzero(subjects == heldout)
        if set(subjects[train_idx]) & set(subjects[test_idx]):
            raise RuntimeError(f"subject leakage in {target} {heldout}")
        for model_name in models:
            try:
                if model_name == "1D-CNN":
                    pred = fit_predict_cnn(seq, y, train_idx, test_idx, epochs=4)
                else:
                    pred = fit_predict_classical(X, y, train_idx, test_idx, model_name)
                append_prediction_rows(pred_rows, dataset_name, target, split_name, model_name, heldout, df, test_idx, y[test_idx], pred)
                fold_rows.append(
                    {
                        "dataset": dataset_name,
                        "target": target,
                        "split": split_name,
                        "model": model_name,
                        "subject_id": heldout,
                        "window_sec": int(df["window_sec"].iloc[0]),
                        **metric_dict(y[test_idx], pred),
                    }
                )
            except Exception as exc:
                fold_rows.append(
                    {
                        "dataset": dataset_name,
                        "target": target,
                        "split": split_name,
                        "model": model_name,
                        "subject_id": heldout,
                        "window_sec": int(df["window_sec"].iloc[0]),
                        "balanced_accuracy": np.nan,
                        "macro_f1": np.nan,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return pd.DataFrame(pred_rows), pd.DataFrame(fold_rows)


def evaluate_random5(features: pd.DataFrame, seq: np.ndarray, target: str, models: list[str], dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = features[target].notna()
    positions = np.flatnonzero(mask.to_numpy())
    df = features.loc[mask].copy().reset_index(drop=True)
    seq = seq[positions]
    if df.empty or df[target].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()
    y = df[target].astype(int).to_numpy()
    X = df[CORE_FEATURE_COLS].to_numpy(dtype=np.float64)
    pred_rows: list[dict] = []
    fold_rows: list[dict] = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        for model_name in models:
            try:
                if model_name == "1D-CNN":
                    pred = fit_predict_cnn(seq, y, train_idx, test_idx, epochs=4)
                else:
                    pred = fit_predict_classical(X, y, train_idx, test_idx, model_name)
                append_prediction_rows(
                    pred_rows,
                    dataset_name,
                    target,
                    "random_window_5fold_leakage_demo",
                    model_name,
                    f"fold{fold}",
                    df,
                    test_idx,
                    y[test_idx],
                    pred,
                )
                fold_rows.append(
                    {
                        "dataset": dataset_name,
                        "target": target,
                        "split": "random_window_5fold_leakage_demo",
                        "model": model_name,
                        "subject_id": f"fold{fold}",
                        "window_sec": int(df["window_sec"].iloc[0]),
                        **metric_dict(y[test_idx], pred),
                    }
                )
            except Exception as exc:
                fold_rows.append(
                    {
                        "dataset": dataset_name,
                        "target": target,
                        "split": "random_window_5fold_leakage_demo",
                        "model": model_name,
                        "subject_id": f"fold{fold}",
                        "window_sec": int(df["window_sec"].iloc[0]),
                        "balanced_accuracy": np.nan,
                        "macro_f1": np.nan,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return pd.DataFrame(pred_rows), pd.DataFrame(fold_rows)


def bootstrap_ci(values: np.ndarray, n: int = BOOTSTRAP_N) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(RANDOM_STATE)
    means = np.empty(n, dtype=np.float64)
    for i in range(n):
        means[i] = float(np.nanmean(rng.choice(arr, size=len(arr), replace=True)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_predictions(pred: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    rows = []
    keys = ["dataset", "target", "split", "model", "window_sec"]
    for key, group in pred.groupby(keys, dropna=False):
        dataset, target, split, model, window_sec = key
        pooled = metric_dict(group["y_true"].to_numpy(int), group["y_pred"].to_numpy(int))
        fold_group = folds[
            (folds["dataset"] == dataset)
            & (folds["target"] == target)
            & (folds["split"] == split)
            & (folds["model"] == model)
            & (folds["window_sec"].astype(int) == int(window_sec))
        ]
        row = {
            "analysis": "classification",
            "dataset": dataset,
            "target": target,
            "split": split,
            "model": model,
            "window_sec": int(window_sec),
            "n_windows": int(len(group)),
            "n_subjects_or_folds": int(fold_group["subject_id"].nunique()),
            "pooled_balanced_accuracy": pooled["balanced_accuracy"],
            "pooled_macro_f1": pooled["macro_f1"],
        }
        for metric in ["balanced_accuracy", "macro_f1"]:
            vals = fold_group[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(vals)
            row[f"subject_{metric}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            row[f"subject_{metric}_sd"] = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else 0.0
            row[f"subject_{metric}_ci_low"] = lo
            row[f"subject_{metric}_ci_high"] = hi
        row["at_chance_by_subject_ba_ci"] = bool(
            np.isfinite(row["subject_balanced_accuracy_ci_low"])
            and row["subject_balanced_accuracy_ci_low"] <= 0.5 <= row["subject_balanced_accuracy_ci_high"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_wilcoxon(
    fold_metrics: pd.DataFrame,
    left_target: str,
    right_target: str,
    window_sec: int,
    split_name: str = "strict_loso",
) -> dict[str, object]:
    left = fold_metrics[
        (fold_metrics["split"] == split_name)
        & (fold_metrics["model"] == "RandomForest")
        & (fold_metrics["target"] == left_target)
        & (fold_metrics["window_sec"].astype(int) == int(window_sec))
    ][["subject_id", "balanced_accuracy"]].rename(columns={"balanced_accuracy": "left_ba"})
    right = fold_metrics[
        (fold_metrics["split"] == split_name)
        & (fold_metrics["model"] == "RandomForest")
        & (fold_metrics["target"] == right_target)
        & (fold_metrics["window_sec"].astype(int) == int(window_sec))
    ][["subject_id", "balanced_accuracy"]].rename(columns={"balanced_accuracy": "right_ba"})
    merged = left.merge(right, on="subject_id", how="inner").dropna()
    if len(merged) == 0:
        stat = np.nan
        p_value = np.nan
    else:
        diff = merged["left_ba"].to_numpy(float) - merged["right_ba"].to_numpy(float)
        try:
            stat, p_value = wilcoxon(diff, alternative="greater", zero_method="zsplit")
            stat = float(stat)
            p_value = float(p_value)
        except Exception:
            stat = np.nan
            p_value = np.nan
    return {
        "analysis": "paired_wilcoxon_rf_same_windows",
        "left_target": left_target,
        "right_target": right_target,
        "window_sec": int(window_sec),
        "n_subjects": int(len(merged)),
        "left_subject_ba_mean": float(merged["left_ba"].mean()) if len(merged) else np.nan,
        "right_subject_ba_mean": float(merged["right_ba"].mean()) if len(merged) else np.nan,
        "mean_difference": float((merged["left_ba"] - merged["right_ba"]).mean()) if len(merged) else np.nan,
        "wilcoxon_stat": stat,
        "wilcoxon_p_greater": p_value,
    }


def duplicate_fingerprint(features: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for subject, group in features.sort_values(["subject_id", "task_id", "start_sec"]).groupby("subject_id"):
        mat = group[["task_id", "start_sec"] + CORE_FEATURE_COLS].copy()
        mat[CORE_FEATURE_COLS] = mat[CORE_FEATURE_COLS].round(8)
        payload = mat.to_csv(index=False).encode("utf-8")
        rows.append({"subject_id": subject, "n_windows": len(group), "fingerprint": hashlib.md5(payload).hexdigest()})
    fp = pd.DataFrame(rows)
    dup = fp[fp.duplicated("fingerprint", keep=False)].sort_values("fingerprint")
    fp["duplicate_group_size"] = fp.groupby("fingerprint")["subject_id"].transform("count")
    fp.to_csv(out_dir / "stressid_duplicate_fingerprints.csv", index=False)
    dup.to_csv(out_dir / "stressid_duplicate_subjects.csv", index=False)
    return dup


def inspect_structure(root: Path, self_report: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    phys_root = root / "Physiological"
    subjects = sorted([p.name for p in phys_root.iterdir() if p.is_dir() and not p.name.startswith("._")])
    file_counts = []
    for subject in subjects:
        files = sorted([p.stem.replace(f"{subject}_", "") for p in (phys_root / subject).glob("*.txt")])
        file_counts.append({"subject_id": subject, "n_physio_txt": len(files), "tasks": "|".join(files)})
    file_counts_df = pd.DataFrame(file_counts)
    file_counts_df.to_csv(out_dir / "stressid_structure_inventory.csv", index=False)
    sample = load_task_physio(root, "2ea4", "Baseline")
    sample_report = {
        "sample_subject": "2ea4",
        "sample_task": "Baseline",
        "sample_shape": str(sample.shape),
        "sample_dtypes": str(sample.dtypes.to_dict()),
        "estimated_ecg_sampling_rate_hz": FS_RAW,
        "estimated_eda_sampling_rate_hz": FS_RAW,
        "estimated_resp_sampling_rate_hz": FS_RAW,
        "rating_scale": "0-10",
        "n_subject_dirs": int(len(subjects)),
        "n_selfreport_subjects": int(self_report["subject_id"].nunique()),
        "n_selfreport_task_rows": int(len(self_report)),
        "selfreport_missing_all_four_rows": int(self_report[["relax", "stress", "valence", "arousal"]].isna().any(axis=1).sum()),
    }
    return sample_report


def make_figure(results: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rf = results[
            (results["analysis"] == "classification")
            & (results["split"] == "strict_loso")
            & (results["model"] == "RandomForest")
            & (results["window_sec"].astype(int) == 20)
            & (results["target"].isin(PRIMARY_TARGETS))
        ].copy()
        if rf.empty:
            return
        order = PRIMARY_TARGETS
        rf["target"] = pd.Categorical(rf["target"], categories=order, ordered=True)
        rf = rf.sort_values("target")
        x = np.arange(len(rf))
        y = rf["subject_balanced_accuracy_mean"].to_numpy(float)
        lo = y - rf["subject_balanced_accuracy_ci_low"].to_numpy(float)
        hi = rf["subject_balanced_accuracy_ci_high"].to_numpy(float) - y
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x, y, color=["#5b7cfa", "#2b9c79", "#c55a4a"], alpha=0.86)
        ax.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt="none", color="black", capsize=4, linewidth=1.2)
        ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(["Task", "Arousal", "Subjective stress"])
        ax.set_ylabel("Subject-level balanced accuracy")
        ax.set_ylim(0, 1)
        ax.set_title("StressID strict LOSO RF by label type")
        fig.tight_layout()
        fig.savefig(out_dir / "stressid_ba_by_label_type.png", dpi=180)
        plt.close(fig)
    except Exception:
        traceback.print_exc()


def run(args: argparse.Namespace) -> None:
    seed_everything()
    root = Path(args.stressid_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.log").write_text("", encoding="utf-8")
    log(f"StressID root={root}", out_dir)
    log(f"Output dir={out_dir}", out_dir)
    self_report = parse_self_assessments(root, out_dir)
    inspection = inspect_structure(root, self_report, out_dir)
    log(f"Inspection: {inspection}", out_dir)

    all_predictions = []
    all_fold_metrics = []
    all_results = []
    paired_rows = []
    feature_cache: dict[int, tuple[pd.DataFrame, np.ndarray]] = {}

    for window_sec in WINDOW_LENGTHS:
        try:
            log(f"Building StressID windows window_sec={window_sec}", out_dir)
            features, seq = build_features(root, out_dir, self_report, window_sec)
            feature_cache[window_sec] = (features, seq)
            log(
                f"Feature rows window_sec={window_sec}: rows={len(features)} subjects={features['subject_id'].nunique()} tasks={features['task_id'].nunique()}",
                out_dir,
            )
            if window_sec == 20:
                dup = duplicate_fingerprint(features, out_dir)
                log(f"Duplicate subject fingerprint groups={dup['fingerprint'].nunique() if not dup.empty else 0}", out_dir)
            if window_sec == 20:
                eval_plan = [(target, MODELS) for target in PRIMARY_TARGETS]
                eval_plan.extend((target, ["RandomForest"]) for target in SECONDARY_TARGETS)
            else:
                eval_plan = []
            for target, model_list in eval_plan:
                log(f"LOSO target={target} window_sec={window_sec}", out_dir)
                pred, folds = evaluate_loso(features, seq, target, model_list, "StressID")
                all_predictions.append(pred)
                all_fold_metrics.append(folds)
                all_results.append(summarize_predictions(pred, folds))
                pd.concat([x for x in all_predictions if not x.empty], ignore_index=True).to_csv(
                    out_dir / "stressid_predictions.partial.csv", index=False
                )
                pd.concat([x for x in all_fold_metrics if not x.empty], ignore_index=True).to_csv(
                    out_dir / "stressid_fold_metrics.partial.csv", index=False
                )
                pd.concat([x for x in all_results if not x.empty], ignore_index=True).to_csv(
                    out_dir / "stressid_results.partial.csv", index=False
                )
            if window_sec == 20:
                log("Random-window 5-fold subjective stress leakage demo", out_dir)
                pred, folds = evaluate_random5(features, seq, "stress_median", MODELS, "StressID")
                all_predictions.append(pred)
                all_fold_metrics.append(folds)
                all_results.append(summarize_predictions(pred, folds))
        except Exception as exc:
            log(f"WINDOW_FAIL window_sec={window_sec}: {type(exc).__name__}: {exc}", out_dir)
            log(traceback.format_exc(), out_dir)

    # Paired tests must use the same rows/windows. Recompute RF-only LOSO on pair-specific common masks.
    paired_fold_metrics = []
    for window_sec, (features, seq) in feature_cache.items():
        for left, right in [("arousal_median", "stress_median"), ("task_condition", "stress_median")]:
            mask = features[left].notna() & features[right].notna()
            split_name = f"strict_loso_paired_{left}_vs_{right}"
            local_pair_folds = []
            for target in [left, right]:
                pred, folds = evaluate_loso(
                    features,
                    seq,
                    target,
                    ["RandomForest"],
                    "StressID",
                    split_name=split_name,
                    common_mask=mask,
                )
                if not pred.empty:
                    all_predictions.append(pred)
                    all_results.append(summarize_predictions(pred, folds))
                if not folds.empty:
                    paired_fold_metrics.append(folds)
                    local_pair_folds.append(folds)
            if local_pair_folds:
                fm = pd.concat(local_pair_folds, ignore_index=True)
                paired_rows.append(paired_wilcoxon(fm, left, right, window_sec, split_name=split_name))

    pred_df = pd.concat([x for x in all_predictions if not x.empty], ignore_index=True) if all_predictions else pd.DataFrame()
    fold_df = pd.concat([x for x in all_fold_metrics + paired_fold_metrics if not x.empty], ignore_index=True) if (all_fold_metrics or paired_fold_metrics) else pd.DataFrame()
    result_df = pd.concat([x for x in all_results if not x.empty], ignore_index=True) if all_results else pd.DataFrame()
    paired_df = pd.DataFrame(paired_rows).drop_duplicates(["left_target", "right_target", "window_sec"], keep="last")
    if not paired_df.empty:
        paired_out = paired_df.copy()
        paired_out["dataset"] = "StressID"
        paired_out["split"] = "strict_loso_paired_same_windows"
        paired_out["model"] = "RandomForest"
        result_df = pd.concat([result_df, paired_out], ignore_index=True, sort=False)

    pred_df.to_csv(out_dir / "stressid_predictions.csv", index=False)
    fold_df.to_csv(out_dir / "stressid_fold_metrics.csv", index=False)
    paired_df.to_csv(out_dir / "stressid_paired_wilcoxon.csv", index=False)
    result_df.to_csv(out_dir / "stressid_results.csv", index=False)
    make_figure(result_df, out_dir)
    write_report(out_dir, inspection, self_report, result_df, paired_df)
    print((out_dir / "stressid_report.md").read_text(encoding="utf-8"), flush=True)


def write_report(out_dir: Path, inspection: dict[str, object], self_report: pd.DataFrame, results: pd.DataFrame, paired: pd.DataFrame) -> None:
    primary = results[
        (results["analysis"] == "classification")
        & (results["split"] == "strict_loso")
        & (results["model"] == "RandomForest")
        & (results["window_sec"].astype("Int64") == 20)
        & (results["target"].isin(PRIMARY_TARGETS))
    ].copy()
    range_rows = []
    strict20 = results[
        (results["analysis"] == "classification")
        & (results["split"] == "strict_loso")
        & (results["window_sec"].astype("Int64") == 20)
        & (results["target"].isin(PRIMARY_TARGETS))
    ]
    for target, group in strict20.groupby("target"):
        rf = group[group["model"] == "RandomForest"].head(1)
        range_rows.append(
            {
                "target": target,
                "rf_subject_ba": rf["subject_balanced_accuracy_mean"].iloc[0] if not rf.empty else np.nan,
                "rf_ci": (
                    f"[{fmt(rf['subject_balanced_accuracy_ci_low'].iloc[0])}, {fmt(rf['subject_balanced_accuracy_ci_high'].iloc[0])}]"
                    if not rf.empty
                    else "NA"
                ),
                "across_model_pooled_ba_range": (
                    f"{fmt(group['pooled_balanced_accuracy'].min())}-{fmt(group['pooled_balanced_accuracy'].max())}"
                    if not group.empty
                    else "NA"
                ),
                "at_chance": bool(rf["at_chance_by_subject_ba_ci"].iloc[0]) if not rf.empty else True,
            }
        )
    range_df = pd.DataFrame(range_rows)
    random_vs_loso = results[
        (results["analysis"] == "classification")
        & (results["target"] == "stress_median")
        & (results["window_sec"].astype("Int64") == 20)
        & (results["split"].isin(["strict_loso", "random_window_5fold_leakage_demo"]))
    ][["target", "split", "model", "pooled_balanced_accuracy", "pooled_macro_f1", "subject_balanced_accuracy_mean"]]
    missing = self_report[["relax", "stress", "valence", "arousal"]].isna().sum().reset_index()
    missing.columns = ["rating", "missing_task_subject_rows"]
    feature_summary = pd.DataFrame()
    failure_summary = pd.DataFrame()
    feature_path = out_dir / "stressid_features.csv"
    failure_path = out_dir / "stressid_feature_failures_20s.csv"
    if feature_path.exists():
        feat = pd.read_csv(feature_path, usecols=["subject_id", "task_id", "task_condition", "arousal_median", "stress_median", "arousal_tertile", "stress_tertile"])
        feature_summary = pd.DataFrame(
            [
                {
                    "target_or_file": "all_20s_windows",
                    "usable_windows": len(feat),
                    "usable_subjects": feat["subject_id"].nunique(),
                    "usable_tasks": feat["task_id"].nunique(),
                },
                *[
                    {
                        "target_or_file": col,
                        "usable_windows": int(feat[col].notna().sum()),
                        "usable_subjects": int(feat.loc[feat[col].notna(), "subject_id"].nunique()),
                        "usable_tasks": int(feat.loc[feat[col].notna(), "task_id"].nunique()),
                    }
                    for col in ["task_condition", "arousal_median", "stress_median", "arousal_tertile", "stress_tertile"]
                ],
            ]
        )
    if failure_path.exists():
        failure_summary = pd.read_csv(failure_path)
    lines = [
        "# StressID Leakage-Free LOSO Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Structure",
        f"- Subject folders: {inspection['n_subject_dirs']}",
        f"- Self-report task rows: {inspection['n_selfreport_task_rows']} across {inspection['n_selfreport_subjects']} subjects.",
        f"- Sample loaded: subject {inspection['sample_subject']} task {inspection['sample_task']} shape {inspection['sample_shape']} dtype {inspection['sample_dtypes']}.",
        "- Physiological columns: ECG, EDA, RR in task-level text files.",
        "- Estimated sampling rate: ECG=500Hz, EDA=500Hz, respiration/RR=500Hz; all are downsampled to 4Hz for window-aligned sequence/EDA processing.",
        "- Rating scale/range in self_assessments.csv: 0-10 for stress, relaxation, arousal, valence.",
        "",
        "Missing self-report values:",
        table_md(missing),
        "",
        "Usable N after intersecting physiological files and labels (20s/5s default):",
        table_md(feature_summary),
        "",
        "Dropped subject-task recordings at 20s/5s:",
        table_md(failure_summary),
        "",
        "## Task Mapping",
        f"- Stressor=1: {', '.join(sorted(STRESSOR_TASKS))}.",
        f"- Non-stressor=0: {', '.join(sorted(NONSTRESSOR_TASKS))}.",
        "- Baseline physiological files are not included in self-report comparisons because StressID self_assessments.csv has no baseline ratings.",
        "",
        "## Strict LOSO Headline (20s/5s, RF)",
        table_md(range_df),
        "",
        "## Paired Wilcoxon (RF, Same Subjects/Windows)",
        table_md(paired.sort_values(["left_target", "window_sec"]) if not paired.empty else paired),
        "",
        "## Random Window Leakage Demo",
        table_md(random_vs_loso.sort_values(["split", "model"]) if not random_vs_loso.empty else random_vs_loso, max_rows=60),
        "",
        "## Integrity",
        "- Duplicate fingerprints are saved in stressid_duplicate_subjects.csv and stressid_duplicate_fingerprints.csv.",
        "",
        "## Outputs",
        "- stressid_features.csv, stressid_selfreport.csv, stressid_results.csv",
        "- stressid_fold_metrics.csv, stressid_predictions.csv, stressid_paired_wilcoxon.csv",
        "- stressid_ba_by_label_type.png",
    ]
    (out_dir / "stressid_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StressID leakage-free wearable-stress analysis.")
    parser.add_argument(
        "--stressid-root", type=Path, required=True,
        help="StressID folder containing Physiological/ and the three label CSV files.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("stressid_run"))
    parser.add_argument(
        "--features-only", action="store_true",
        help="Write the 20-second features and sequences used by the manuscript.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parse_args()
    if args.features_only:
        required_inputs = (
            "Physiological", "self_assessments.csv", "labels.csv", "labels_supplementary.csv",
        )
        missing = [name for name in required_inputs if not (args.stressid_root / name).exists()]
        if missing:
            raise SystemExit(f"Missing StressID inputs: {', '.join(missing)}")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        self_report = parse_self_assessments(args.stressid_root, args.out_dir)
        features, _ = build_features(args.stressid_root, args.out_dir, self_report, 20)
        print(f"Wrote {len(features):,} windows from {features.subject_id.nunique()} participants.")
    else:
        run(args)
