from __future__ import annotations

import io
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np
import pandas as pd


TASKS = [f"T{i}" for i in range(1, 10)]
FEATURE_TASKS = [f"T{i}" for i in range(1, 9)]

EMOTION_COLS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
AU_COLS = [
    "lip_puckerer",
    "lower_lip_depressor",
    "upper_lip_raiser",
    "mouth_stretch",
    "lip_tightener",
    "upper_lid_raiser",
    "cheek_raiser",
]
BIO_SIGNAL_COLS = ["eda", "hr", "temp"]
SEQUENCE_COLS = EMOTION_COLS + BIO_SIGNAL_COLS

XCEPTION_FEATURE_COLS = [
    "angry_mean",
    "disgust_mean",
    "scared_mean",
    "happy_mean",
    "sad_mean",
    "surprised_mean",
    "neutral_mean",
]

AU_FEATURE_COLS = [
    "lip_puckerer_mean",
    "lower_lip_depressor_mean",
    "upper_lip_raiser_mean",
    "mouth_stretch_mean",
    "lip_tightener_mean",
    "upper_lid_raiser_mean",
    "cheek_raiser_mean",
]

BIO_FEATURE_COLS = [
    "eda_mean",
    "eda_min",
    "eda_max",
    "eda_std",
    "eda_kurtosis",
    "eda_skew",
    "eda_num_peaks",
    "eda_peak_amplitude",
    "eda_peak_duration",
    "hr_mean",
    "hr_min",
    "hr_max",
    "hr_std",
    "hr_rms",
    "hr_num_peaks",
    "hr_peak_amplitude",
    "hr_peak_duration",
    "temp_mean",
    "temp_min",
    "temp_max",
    "temp_std",
]

FEATURE_COLS = XCEPTION_FEATURE_COLS + BIO_FEATURE_COLS
AU_ALL_FEATURE_COLS = AU_FEATURE_COLS + BIO_FEATURE_COLS
BIOMETRIC_FEATURE_COLS = [c for c in FEATURE_COLS if c.startswith(("eda_", "hr_", "temp_"))]

TASK_LABEL_NUMERIC = {
    "T1": 0,
    "T2": 1,
    "T3": 1,
    "T4": 0,
    "T5": 1,
    "T6": 0,
    "T7": 2,
    "T8": 0,
    "T9": np.nan,
}

TASK_LABEL_NAME = {
    "T1": "non_stressor",
    "T2": "stressor",
    "T3": "stressor",
    "T4": "non_stressor",
    "T5": "stressor",
    "T6": "non_stressor",
    "T7": "amusement",
    "T8": "non_stressor",
    "T9": "recovery",
}

STRESS_INTERVAL_COLS = [
    "Stress_first_interval",
    "Stress_second_interval",
    "Stress_third_interval",
    "Stress_forth_interval_",
    "Stress_fourth_interval",
    "Stress_fifth_interval",
]

NASA_COLS = [
    "Mental Demand",
    "Physical demand",
    "Temporal demand",
    "Performance",
    "Effort",
    "Frustration",
]


def subject_sort_key(path_or_id: str | Path) -> int:
    text = Path(path_or_id).stem if isinstance(path_or_id, Path) else str(path_or_id)
    match = re.search(r"S(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else 10_000


def subject_id_from_path(path: Path) -> str:
    return f"S{subject_sort_key(path)}"


def site_for_subject(subject_id: str) -> str:
    return "Finland" if subject_sort_key(subject_id) <= 6 else "U.S."


def list_subject_archives(data_root: Path) -> list[Path]:
    archives = sorted(data_root.glob("S*.zip"), key=subject_sort_key)
    if archives:
        return archives
    return sorted(data_root.rglob("S*.zip"), key=subject_sort_key)


def task_from_entry(entry_name: str) -> str | None:
    path = PurePosixPath(entry_name)
    for part in path.parts:
        match = re.fullmatch(r"[Tt]([1-9])", part)
        if match:
            return f"T{match.group(1)}"
    fname = path.name
    match = re.search(r"(?<![A-Za-z0-9])[Tt]([1-9])(?![0-9])", fname)
    if match:
        return f"T{match.group(1)}"
    match = re.search(r"[A-Za-z0-9]*[Tt]([1-9])(?:XCEPTION|xception)", fname)
    if match:
        return f"T{match.group(1)}"
    return None


def is_hidden_or_inner_archive(name: str) -> bool:
    path = PurePosixPath(name)
    return any(part.startswith(".") for part in path.parts) or path.suffix.lower() == ".zip"


@dataclass
class TaskFiles:
    subject_id: str
    task_id: str
    site: str
    archive_path: str
    xception_path: str | None = None
    hr_path: str | None = None
    eda_path: str | None = None
    temp_path: str | None = None
    acc_path: str | None = None
    bvp_path: str | None = None
    ibi_path: str | None = None
    landmarks_path: str | None = None
    video_path: str | None = None
    label_path: str | None = None

    @property
    def has_video(self) -> bool:
        return self.video_path is not None

    @property
    def has_label_file(self) -> bool:
        return self.label_path is not None


def is_label_workbook(name: str) -> bool:
    path = PurePosixPath(name)
    fname = path.name.lower()
    if not fname.endswith((".xlsx", ".xls")):
        return False
    if "xception" in fname:
        return False
    return task_from_entry(name) is None


def discover_task_files(zip_path: Path) -> dict[str, TaskFiles]:
    subject_id = subject_id_from_path(zip_path)
    task_files = {
        task: TaskFiles(
            subject_id=subject_id,
            task_id=task,
            site=site_for_subject(subject_id),
            archive_path=str(zip_path),
        )
        for task in TASKS
    }
    label_path = None
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not is_hidden_or_inner_archive(n)]
        label_candidates = [n for n in names if is_label_workbook(n)]
        if label_candidates:
            label_path = sorted(label_candidates, key=lambda n: (n.count("/"), len(n), n))[0]

        for name in names:
            task = task_from_entry(name)
            if task not in task_files:
                continue
            fname = PurePosixPath(name).name.lower()
            suffix = PurePosixPath(name).suffix.lower()
            files = task_files[task]
            if suffix in {".xlsx", ".xls"}:
                if "xception" in fname or re.fullmatch(r"[a-z0-9]*t[1-9]\.xlsx", fname):
                    files.xception_path = choose_best(files.xception_path, name)
            elif suffix == ".csv":
                stem = PurePosixPath(name).stem.lower()
                if stem.endswith("hr") or stem == "hr":
                    files.hr_path = choose_best(files.hr_path, name)
                elif stem.endswith("eda") or stem == "eda":
                    files.eda_path = choose_best(files.eda_path, name)
                elif stem.endswith("temp") or stem == "temp":
                    files.temp_path = choose_best(files.temp_path, name)
                elif stem.endswith("acc") or stem == "acc":
                    files.acc_path = choose_best(files.acc_path, name)
                elif stem.endswith("bvp") or stem == "bvp":
                    files.bvp_path = choose_best(files.bvp_path, name)
                elif stem.endswith("ibi") or stem == "ibi":
                    files.ibi_path = choose_best(files.ibi_path, name)
                elif "stress" in stem and label_path is None:
                    label_path = name
            elif suffix == ".json" and "landmark" in fname:
                files.landmarks_path = choose_best(files.landmarks_path, name)
            elif suffix in {".mp4", ".avi", ".mov"}:
                files.video_path = choose_best(files.video_path, name)

    for files in task_files.values():
        files.label_path = label_path
    return task_files


def choose_best(current: str | None, candidate: str) -> str:
    if current is None:
        return candidate
    current_score = (current.count("/"), len(current), current.lower())
    candidate_score = (candidate.count("/"), len(candidate), candidate.lower())
    return candidate if candidate_score < current_score else current


def zip_read_bytes(zip_path: str | Path, member: str) -> bytes:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.read(member)


def read_excel_member(zip_path: str | Path, member: str, **kwargs) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(zip_read_bytes(zip_path, member)), **kwargs)


def read_csv_member(zip_path: str | Path, member: str, **kwargs) -> pd.DataFrame:
    data = zip_read_bytes(zip_path, member)
    if data.startswith(b"\xd0\xcf\x11\xe0"):
        return pd.read_excel(io.BytesIO(data), **kwargs)
    return pd.read_csv(io.BytesIO(data), **kwargs)


def e4_metadata(zip_path: str | Path, member: str | None) -> dict[str, float | int | None]:
    if member is None:
        return {"start_time": None, "sampling_rate": None, "sample_count": None, "duration_sec": None}
    try:
        raw = read_csv_member(zip_path, member, header=None)
        if raw.empty:
            return {"start_time": None, "sampling_rate": None, "sample_count": 0, "duration_sec": None}
        fmt = detect_signal_format(raw, member)
        sampling_rate = fmt["sampling_rate"]
        sample_count = int(max(raw.shape[0] - fmt["data_start_row"], 0))
        duration = sample_count / sampling_rate if sampling_rate and sampling_rate > 0 else None
        return {
            "start_time": fmt["start_time"],
            "sampling_rate": sampling_rate,
            "sample_count": sample_count,
            "duration_sec": duration,
        }
    except Exception:
        return {"start_time": None, "sampling_rate": None, "sample_count": None, "duration_sec": None}


def xception_metadata(zip_path: str | Path, member: str | None) -> dict[str, float | int | None]:
    if member is None:
        return {"start_time": 0.0, "sampling_rate": None, "sample_count": None, "duration_sec": None}
    try:
        df = read_excel_member(zip_path, member)
        if df.empty:
            return {"start_time": 0.0, "sampling_rate": None, "sample_count": 0, "duration_sec": None}
        time_col = _find_time_column(df)
        if time_col is None:
            return {"start_time": 0.0, "sampling_rate": None, "sample_count": int(len(df)), "duration_sec": None}
        times = pd.to_numeric(df[time_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(times) < 2:
            return {"start_time": 0.0, "sampling_rate": None, "sample_count": int(len(times)), "duration_sec": None}
        rel_times = times - times[0] if time_col.lower() == "timestamp" else times
        diffs = np.diff(rel_times)
        diffs = diffs[diffs > 0]
        sr = 1.0 / float(np.median(diffs)) if len(diffs) else None
        duration = float(np.nanmax(rel_times) - np.nanmin(rel_times)) if len(rel_times) else None
        return {"start_time": 0.0, "sampling_rate": sr, "sample_count": int(len(times)), "duration_sec": duration}
    except Exception:
        return {"start_time": 0.0, "sampling_rate": None, "sample_count": None, "duration_sec": None}


def _find_time_column(df: pd.DataFrame) -> str | None:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for key in ["time (seconds)", "timestamp", "time", "seconds"]:
        if key in normalized:
            return normalized[key]
    return None


def load_e4_signal(zip_path: str | Path, member: str, value_name: str) -> tuple[pd.DataFrame, float, float]:
    raw = read_csv_member(zip_path, member, header=None)
    if raw.empty:
        raise ValueError(f"{member} is empty")
    fmt = detect_signal_format(raw, member, value_name=value_name)
    sampling_rate = fmt["sampling_rate"]
    if sampling_rate is None or sampling_rate <= 0:
        raise ValueError(f"{member} lacks an inferable sampling rate")
    start_time = 0.0 if fmt["start_time"] is None else float(fmt["start_time"])
    values = raw.iloc[fmt["data_start_row"] :].apply(pd.to_numeric, errors="coerce")
    value_series = select_signal_value_column(values, value_name)
    values = pd.DataFrame({value_name: clean_physiology_values(value_series, value_name)})
    values = values.dropna(subset=[value_name]).reset_index(drop=True)
    values["time_abs"] = start_time + np.arange(len(values), dtype=float) / sampling_rate
    return values[["time_abs", value_name]], start_time, sampling_rate


def select_signal_value_column(values: pd.DataFrame, value_name: str) -> pd.Series:
    if values.shape[1] == 1:
        return values.iloc[:, 0]
    numeric = values.apply(pd.to_numeric, errors="coerce")
    epoch_like_fraction = numeric.apply(lambda col: (col > 1_000_000_000).mean())
    candidates = [col for col in numeric.columns if epoch_like_fraction[col] < 0.5]
    if candidates:
        return numeric[candidates[0]]
    return numeric.mean(axis=1, skipna=True)


def clean_physiology_values(series: pd.Series, value_name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    ranges = {
        "hr": (20.0, 240.0),
        "eda": (-0.01, 100.0),
        "temp": (15.0, 45.0),
        "bvp": (-10000.0, 10000.0),
    }
    lo_hi = ranges.get(value_name.lower())
    if lo_hi:
        lo, hi = lo_hi
        values = values.mask((values < lo) | (values > hi))
    return values.interpolate(limit_direction="both")


def infer_signal_name(member: str, value_name: str | None = None) -> str | None:
    if value_name:
        return value_name.lower()
    stem = PurePosixPath(member).stem.lower()
    for signal in ["hr", "eda", "temp", "acc", "bvp", "ibi"]:
        if stem.endswith(signal) or stem == signal:
            return signal
    return None


def expected_sampling_rate(signal: str | None) -> float | None:
    return {
        "hr": 1.0,
        "eda": 4.0,
        "temp": 4.0,
        "acc": 32.0,
        "bvp": 64.0,
    }.get(signal or "")


def detect_signal_format(raw: pd.DataFrame, member: str, value_name: str | None = None) -> dict[str, float | int | None]:
    signal = infer_signal_name(member, value_name=value_name)
    expected_rate = expected_sampling_rate(signal)
    first = pd.to_numeric(raw.iloc[0], errors="coerce").dropna()
    second = pd.to_numeric(raw.iloc[1], errors="coerce").dropna() if raw.shape[0] > 1 else pd.Series(dtype=float)
    first_value = float(first.iloc[0]) if not first.empty else None
    second_value = float(second.iloc[0]) if not second.empty else None
    first_is_epoch = first_value is not None and first_value > 1_000_000_000
    second_is_declared_rate = (
        second_value is not None
        and expected_rate is not None
        and abs(second_value - expected_rate) < 1e-6
    )
    if first_is_epoch and second_is_declared_rate:
        return {"start_time": first_value, "sampling_rate": second_value, "data_start_row": 2}
    if first_is_epoch:
        return {"start_time": first_value, "sampling_rate": expected_rate, "data_start_row": 1}
    return {"start_time": None, "sampling_rate": expected_rate, "data_start_row": 0}


def load_emotion_signal(zip_path: str | Path, member: str) -> pd.DataFrame:
    df = read_excel_member(zip_path, member)
    if df.empty:
        raise ValueError(f"{member} is empty")
    time_col = _find_time_column(df)
    if time_col is None:
        raise ValueError(f"{member} lacks a time column")
    out = pd.DataFrame()
    times = pd.to_numeric(df[time_col], errors="coerce")
    if str(time_col).strip().lower() == "timestamp" and times.notna().any():
        times = times - float(times.dropna().iloc[0])
    out["time_rel"] = times

    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    if "emotion" in lower_cols and "score" in lower_cols:
        for col in EMOTION_COLS:
            out[col] = 0.0
        emotions = df[lower_cols["emotion"]].astype(str).str.strip().str.lower()
        scores = pd.to_numeric(df[lower_cols["score"]], errors="coerce").fillna(0.0)
        aliases = {"fear": "scared", "scare": "scared", "surprise": "surprised"}
        for idx, emo in emotions.items():
            key = aliases.get(emo, emo)
            if key in EMOTION_COLS:
                out.at[idx, key] = float(scores.at[idx])
    else:
        aliases = {
            "angry": "angry",
            "disgust": "disgust",
            "scared": "scared",
            "fear": "scared",
            "happy": "happy",
            "sad": "sad",
            "surprised": "surprised",
            "surprise": "surprised",
            "neutral": "neutral",
        }
        for canonical in EMOTION_COLS:
            out[canonical] = np.nan
        for source, canonical in aliases.items():
            if source in lower_cols:
                out[canonical] = pd.to_numeric(df[lower_cols[source]], errors="coerce")
    out = out.dropna(subset=["time_rel"]).sort_values("time_rel")
    for col in EMOTION_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out[["time_rel"] + EMOTION_COLS].reset_index(drop=True)


def resample_to_grid(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    mask = np.isfinite(times) & np.isfinite(values)
    times = times[mask]
    values = values[mask]
    if len(times) == 0:
        return np.full(len(grid), np.nan)
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    unique_times, unique_idx = np.unique(times, return_index=True)
    unique_values = values[unique_idx]
    if len(unique_times) == 1:
        return np.full(len(grid), unique_values[0], dtype=float)
    return np.interp(grid, unique_times, unique_values, left=unique_values[0], right=unique_values[-1])


def load_label_sheet(zip_path: str | Path, label_member: str | None) -> pd.DataFrame | None:
    if label_member is None:
        return None
    if label_member.lower().endswith(".csv"):
        return read_csv_member(zip_path, label_member)
    return read_excel_member(zip_path, label_member)


def row_for_task(label_df: pd.DataFrame | None, task_id: str) -> pd.Series | None:
    if label_df is None or label_df.empty or task_id not in {"T2", "T3", "T5", "T7"}:
        return None
    task_col = None
    for col in label_df.columns:
        if str(col).strip().lower() == "task":
            task_col = col
            break
    if task_col is not None:
        task_text = label_df[task_col].astype(str).str.strip().str.lower()
        if task_id == "T2":
            mask = task_text.str.contains("prepar", na=False)
        elif task_id == "T3":
            mask = task_text.str.contains("presentation", na=False) & ~task_text.str.contains("prepar", na=False)
        elif task_id == "T5":
            mask = task_text.str.contains("iq|stroop|test", regex=True, na=False)
        else:
            mask = task_text.str.contains("watch|video", regex=True, na=False) & ~task_text.str.contains("no video", na=False)
        if mask.any():
            return label_df.loc[mask].iloc[0]

    fallback_index = {"T2": 0, "T3": 1, "T5": 2, "T7": 3}.get(task_id)
    if fallback_index is not None and fallback_index < len(label_df):
        return label_df.iloc[fallback_index]
    return None


def stress_intervals_for_task(label_df: pd.DataFrame | None, task_id: str) -> list[float] | None:
    if task_id in {"T1", "T4", "T6", "T8"}:
        return [0.0] * 5
    row = row_for_task(label_df, task_id)
    if row is None:
        return None
    values = []
    for col in STRESS_INTERVAL_COLS:
        if col in row.index:
            val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.notna(val):
                values.append(float(val))
    if len(values) < 5:
        stress_cols = [c for c in row.index if str(c).lower().startswith("stress")]
        values = []
        for col in stress_cols[:5]:
            val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.notna(val):
                values.append(float(val))
    if not values:
        return None
    while len(values) < 5:
        values.append(values[-1])
    values = values[:5]
    if max(values) > 2.0:
        values = [v / 10.0 for v in values]
    return values


def stress_value_for_window(intervals: list[float] | None, start_sec: float, end_sec: float) -> float | None:
    if intervals is None:
        return None
    sample_times = np.arange(start_sec, end_sec, 0.25)
    if len(sample_times) == 0:
        sample_times = np.array([(start_sec + end_sec) / 2.0])
    # Paper: five task sub-intervals; first and last span 90s, middle three span 120s.
    boundaries = np.array([0.0, 90.0, 210.0, 330.0, 450.0, math.inf])
    vals = np.array(intervals, dtype=float)
    idx = np.searchsorted(boundaries[1:], sample_times, side="right")
    idx = np.clip(idx, 0, len(vals) - 1)
    return float(np.nanmean(vals[idx]))


def stress_class(stress_value: float | None) -> float:
    if stress_value is None or not np.isfinite(stress_value):
        return np.nan
    if stress_value <= 0.65:
        return 0
    if stress_value <= 1.3:
        return 1
    return 2


def nasa_tlx_for_task(label_df: pd.DataFrame | None, task_id: str) -> float | None:
    if task_id not in {"T2", "T3", "T5"}:
        return None
    row = row_for_task(label_df, task_id)
    if row is None:
        return None
    values = []
    lower_to_col = {str(c).strip().lower(): c for c in row.index}
    for wanted in NASA_COLS:
        col = lower_to_col.get(wanted.lower())
        if col is not None:
            val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.notna(val):
                values.append(float(val))
    if not values:
        return None
    return float(np.mean(values))


def window_count_from_duration(duration_sec: float | None, window_sec: float = 20.0, step_sec: float = 5.0) -> int:
    if duration_sec is None or not np.isfinite(duration_sec) or duration_sec < window_sec:
        return 0
    return int(math.floor((duration_sec - window_sec) / step_sec) + 1)


def print_table(df: pd.DataFrame, max_rows: int = 30) -> None:
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", 80, "display.width", 180):
        print(df.to_string(index=False))
