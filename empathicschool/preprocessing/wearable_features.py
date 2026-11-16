"""Archived 4-Hz EmpathicSchool wearable feature helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from empathic_common import (AU_FEATURE_COLS,
    TASK_LABEL_NAME, TASK_LABEL_NUMERIC, load_e4_signal, read_csv_member)

from empathic_common import resample_to_grid, stress_class, stress_intervals_for_task, stress_value_for_window

import logging

logger = logging.getLogger(__name__)

def log_exception(context, exc):
    logger.warning("%s: %s: %s", context, type(exc).__name__, exc)


TARGET_HZ = 4.0


WINDOW_SEC = 20.0


STEP_SEC = 5.0


WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_HZ)


STEP_SAMPLES = int(STEP_SEC * TARGET_HZ)


PHYSIO_FEATURE_COLS = [
    "eda_scr_peak_count",
    "eda_scr_amp_mean",
    "eda_scr_amp_max",
    "eda_scr_rise_time_mean",
    "eda_phasic_auc",
    "eda_slope",
    "eda_tonic_mean",
    "hrv_mean_ibi",
    "hrv_rmssd",
    "hrv_sdnn",
    "hrv_pnn50",
    "temp_mean",
    "temp_slope",
    "temp_max_drop",
    "acc_mag_mean",
    "acc_mag_std",
    "acc_mag_max",
]


ACC_FEATURE_COLS = ["acc_mag_mean", "acc_mag_std", "acc_mag_max"]


PHYSIO_SEQUENCE_COLS = ["eda", "eda_phasic", "eda_tonic", "ibi_ms", "temp", "acc_mag"]


MULTIMODAL_SEQUENCE_COLS = PHYSIO_SEQUENCE_COLS + AU_FEATURE_COLS


def finite(value: float, default: float = 0.0) -> float:
    return float(value) if np.isfinite(value) else default


def slope(values: np.ndarray, hz: float = TARGET_HZ) -> float:
    arr = pd.Series(values).interpolate(limit_direction="both").to_numpy(dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)[mask] / hz
    y = arr[mask]
    return finite(np.polyfit(x, y, 1)[0])


def max_drop(values: np.ndarray) -> float:
    arr = pd.Series(values).interpolate(limit_direction="both").to_numpy(dtype=float)
    if len(arr) == 0 or not np.isfinite(arr).any():
        return 0.0
    running_max = np.maximum.accumulate(arr)
    return finite(np.nanmax(running_max - arr))


def numeric_clean(values: np.ndarray) -> np.ndarray:
    arr = pd.Series(values).replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").to_numpy(dtype=float)
    if np.isnan(arr).all():
        return np.zeros_like(arr, dtype=float)
    return np.nan_to_num(arr, nan=float(np.nanmedian(arr)))


def decompose_eda(eda: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    clean = numeric_clean(eda)
    try:
        import neurokit2 as nk

        eda_clean = nk.eda_clean(clean, sampling_rate=int(TARGET_HZ))
        comp = nk.eda_phasic(eda_clean, sampling_rate=int(TARGET_HZ))
        phasic = np.asarray(comp["EDA_Phasic"], dtype=float)
        tonic = np.asarray(comp["EDA_Tonic"], dtype=float)
        return numeric_clean(phasic), numeric_clean(tonic), "neurokit2"
    except Exception:
        nyq = 0.5 * TARGET_HZ
        b, a = butter(2, 0.05 / nyq, btype="highpass")
        phasic = filtfilt(b, a, clean) if len(clean) > 16 else clean - np.nanmean(clean)
        tonic = clean - phasic
        return numeric_clean(phasic), numeric_clean(tonic), "butterworth_highpass"


def eda_peak_features(phasic: np.ndarray) -> tuple[int, float, float, float]:
    arr = numeric_clean(phasic)
    if len(arr) < 5:
        return 0, 0.0, 0.0, 0.0
    prominence = max(0.005, float(np.nanstd(arr)) * 0.5)
    peaks, props = find_peaks(arr, distance=int(TARGET_HZ), prominence=prominence)
    if len(peaks) == 0:
        return 0, 0.0, 0.0, 0.0
    prominences = np.asarray(props.get("prominences", np.zeros(len(peaks))), dtype=float)
    left_bases = np.asarray(props.get("left_bases", peaks), dtype=float)
    rise_times = (peaks.astype(float) - left_bases) / TARGET_HZ
    return int(len(peaks)), finite(np.nanmean(prominences)), finite(np.nanmax(prominences)), finite(np.nanmean(rise_times))


def hrv_features(ibi_ms: np.ndarray) -> tuple[float, float, float, float]:
    arr = numeric_clean(ibi_ms)
    arr = arr[np.isfinite(arr) & (arr > 250.0) & (arr < 2500.0)]
    if len(arr) < 2:
        mean_ibi = finite(np.nanmean(arr)) if len(arr) else 0.0
        return mean_ibi, 0.0, 0.0, 0.0
    diffs = np.diff(arr)
    return (
        finite(np.nanmean(arr)),
        finite(np.sqrt(np.nanmean(np.square(diffs)))),
        finite(np.nanstd(arr, ddof=1)),
        finite(np.mean(np.abs(diffs) > 50.0)),
    )


def load_acc_signal(zip_path: Path, member: str) -> tuple[pd.DataFrame, float, float]:
    raw = read_csv_member(zip_path, member, header=None)
    if raw.empty:
        raise ValueError(f"{member} is empty")
    first = pd.to_numeric(raw.iloc[0], errors="coerce").dropna()
    second = pd.to_numeric(raw.iloc[1], errors="coerce").dropna() if len(raw) > 1 else pd.Series(dtype=float)
    start = float(first.iloc[0]) if not first.empty and float(first.iloc[0]) > 1_000_000_000 else 0.0
    rate = 32.0
    data_start = 1
    if not second.empty and abs(float(second.iloc[0]) - 32.0) < 1e-6:
        data_start = 2
    values = raw.iloc[data_start:, :3].apply(pd.to_numeric, errors="coerce")
    values = values.interpolate(limit_direction="both").fillna(0.0)
    values.columns = ["acc_x", "acc_y", "acc_z"]
    values["time_abs"] = start + np.arange(len(values), dtype=float) / rate
    return values[["time_abs", "acc_x", "acc_y", "acc_z"]], start, rate


def load_ibi_signal(zip_path: Path, member: str) -> tuple[pd.DataFrame, float]:
    raw = read_csv_member(zip_path, member, header=None)
    if raw.empty:
        raise ValueError(f"{member} is empty")
    first = pd.to_numeric(raw.iloc[0], errors="coerce").dropna()
    start = float(first.iloc[0]) if not first.empty and float(first.iloc[0]) > 1_000_000_000 else 0.0
    data = raw.iloc[1:].apply(pd.to_numeric, errors="coerce")
    if data.shape[1] >= 2:
        offsets = data.iloc[:, 0].to_numpy(dtype=float)
        ibi = data.iloc[:, 1].to_numpy(dtype=float)
        times = start + offsets
    else:
        ibi = data.iloc[:, 0].to_numpy(dtype=float)
        times = start + np.arange(len(ibi), dtype=float)
    ibi_ms = np.where(ibi < 10.0, ibi * 1000.0, ibi)
    out = pd.DataFrame({"time_abs": times, "ibi_ms": ibi_ms}).dropna()
    return out, start


def ibi_from_bvp(zip_path: Path, member: str) -> tuple[pd.DataFrame, float]:
    bvp, start, sr = load_e4_signal(zip_path, member, "bvp")
    values = numeric_clean(bvp["bvp"].to_numpy(dtype=float))
    peaks, _ = find_peaks(values, distance=max(1, int(sr * 0.35)), prominence=max(0.01, np.std(values) * 0.25))
    if len(peaks) < 2:
        raise ValueError(f"{member} did not yield enough BVP peaks")
    peak_times = bvp["time_abs"].to_numpy(dtype=float)[peaks]
    ibi_ms = np.diff(peak_times) * 1000.0
    return pd.DataFrame({"time_abs": peak_times[1:], "ibi_ms": ibi_ms}), start


def ibi_from_hr(hr: pd.DataFrame) -> pd.DataFrame:
    vals = numeric_clean(hr["hr"].to_numpy(dtype=float))
    ibi_ms = np.where(vals > 0, 60000.0 / vals, np.nan)
    return pd.DataFrame({"time_abs": hr["time_abs"].to_numpy(dtype=float), "ibi_ms": ibi_ms}).dropna()


def task_label_name(task_id: str) -> str:
    return TASK_LABEL_NAME.get(task_id, "")


def task_label_numeric(task_id: str) -> float:
    return TASK_LABEL_NUMERIC.get(task_id, np.nan)


def signal_rel(frame: pd.DataFrame, value_cols: list[str], task_start: float) -> pd.DataFrame:
    out = frame.copy()
    out["time_rel"] = out["time_abs"].to_numpy(dtype=float) - task_start
    return out[["time_rel"] + value_cols]


def load_task_timeseries(archive: Path, files) -> tuple[pd.DataFrame | None, dict]:
    if not files.eda_path or not files.temp_path or not files.acc_path:
        return None, {"reason": "missing eda/temp/acc", "hrv_source": "none", "eda_method": ""}
    try:
        eda, eda_start, _ = load_e4_signal(archive, files.eda_path, "eda")
        temp, temp_start, _ = load_e4_signal(archive, files.temp_path, "temp")
        acc, acc_start, _ = load_acc_signal(archive, files.acc_path)

        hrv_source = "none"
        ibi = None
        if files.ibi_path:
            try:
                ibi, ibi_start = load_ibi_signal(archive, files.ibi_path)
                hrv_source = "ibi"
            except Exception as exc:
                log_exception(f"IBI load fallback {files.subject_id} {files.task_id}", exc)
        if ibi is None and files.bvp_path:
            try:
                ibi, ibi_start = ibi_from_bvp(archive, files.bvp_path)
                hrv_source = "bvp_peaks"
            except Exception as exc:
                log_exception(f"BVP peak fallback {files.subject_id} {files.task_id}", exc)
        if ibi is None and files.hr_path:
            hr, hr_start, _ = load_e4_signal(archive, files.hr_path, "hr")
            ibi = ibi_from_hr(hr)
            ibi_start = hr_start
            hrv_source = "hr_to_ibi"
        if ibi is None or ibi.empty:
            return None, {"reason": "missing ibi/bvp/hr", "hrv_source": "none", "eda_method": ""}

        task_start = min(
            float(eda_start),
            float(temp_start),
            float(acc_start),
            float(ibi["time_abs"].min()),
        )
        frames = [
            signal_rel(eda, ["eda"], task_start),
            signal_rel(temp, ["temp"], task_start),
            signal_rel(acc, ["acc_x", "acc_y", "acc_z"], task_start),
            signal_rel(ibi, ["ibi_ms"], task_start),
        ]
        duration = min(float(frame["time_rel"].max()) for frame in frames if len(frame))
        if files.task_id != "T3" and duration > 540.0:
            duration = 540.0
        if not np.isfinite(duration) or duration < WINDOW_SEC:
            return None, {"reason": "duration shorter than 20s", "hrv_source": hrv_source, "eda_method": ""}

        grid = np.arange(0.0, duration, 1.0 / TARGET_HZ)
        if len(grid) < WINDOW_SAMPLES:
            return None, {"reason": "not enough 4Hz samples", "hrv_source": hrv_source, "eda_method": ""}

        out = pd.DataFrame({"time_rel": grid})
        out["eda"] = resample_to_grid(frames[0]["time_rel"].to_numpy(float), frames[0]["eda"].to_numpy(float), grid)
        out["temp"] = resample_to_grid(frames[1]["time_rel"].to_numpy(float), frames[1]["temp"].to_numpy(float), grid)
        for col in ["acc_x", "acc_y", "acc_z"]:
            out[col] = resample_to_grid(frames[2]["time_rel"].to_numpy(float), frames[2][col].to_numpy(float), grid)
        out["ibi_ms"] = resample_to_grid(frames[3]["time_rel"].to_numpy(float), frames[3]["ibi_ms"].to_numpy(float), grid)
        out["acc_mag"] = np.sqrt(out["acc_x"] ** 2 + out["acc_y"] ** 2 + out["acc_z"] ** 2)
        phasic, tonic, eda_method = decompose_eda(out["eda"].to_numpy(float))
        out["eda_phasic"] = phasic
        out["eda_tonic"] = tonic
        return out, {"reason": "", "hrv_source": hrv_source, "eda_method": eda_method}
    except Exception as exc:
        log_exception(f"load task {files.subject_id} {files.task_id}", exc)
        return None, {"reason": str(exc), "hrv_source": "error", "eda_method": ""}


def window_features(win: pd.DataFrame) -> dict[str, float]:
    phasic = win["eda_phasic"].to_numpy(float)
    eda = win["eda"].to_numpy(float)
    temp = win["temp"].to_numpy(float)
    acc = win["acc_mag"].to_numpy(float)
    ibi = win["ibi_ms"].to_numpy(float)
    peak_count, amp_mean, amp_max, rise_mean = eda_peak_features(phasic)
    mean_ibi, rmssd, sdnn, pnn50 = hrv_features(ibi)
    return {
        "eda_scr_peak_count": float(peak_count),
        "eda_scr_amp_mean": amp_mean,
        "eda_scr_amp_max": amp_max,
        "eda_scr_rise_time_mean": rise_mean,
        "eda_phasic_auc": finite(np.trapezoid(np.maximum(phasic, 0.0), dx=1.0 / TARGET_HZ)),
        "eda_slope": slope(eda),
        "eda_tonic_mean": finite(np.nanmean(win["eda_tonic"].to_numpy(float))),
        "hrv_mean_ibi": mean_ibi,
        "hrv_rmssd": rmssd,
        "hrv_sdnn": sdnn,
        "hrv_pnn50": pnn50,
        "temp_mean": finite(np.nanmean(temp)),
        "temp_slope": slope(temp),
        "temp_max_drop": max_drop(temp),
        "acc_mag_mean": finite(np.nanmean(acc)),
        "acc_mag_std": finite(np.nanstd(acc)),
        "acc_mag_max": finite(np.nanmax(acc)),
    }


def label_values(label_df: pd.DataFrame | None, subject_id: str, task_id: str, start_sec: float,
                 end_sec: float, excluded_subjects: set[str]) -> dict:
    intervals = stress_intervals_for_task(label_df, task_id)
    stress_value = stress_value_for_window(intervals, start_sec, end_sec)
    label = stress_class(stress_value)
    if not np.isfinite(label):
        label = np.nan
    return {
        "L_self_value": stress_value if stress_value is not None else np.nan,
        "L_self": label,
        "L_task": task_label_numeric(task_id),
        "L_task_name": task_label_name(task_id),
        "is_labeled_analysis": subject_id not in excluded_subjects and np.isfinite(label),
    }
