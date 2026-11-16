from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

from empathic_common import (
    AU_COLS,
    AU_ALL_FEATURE_COLS,
    AU_FEATURE_COLS,
    FEATURE_COLS,
    BIO_FEATURE_COLS,
    EMOTION_COLS,
    FEATURE_TASKS,
    XCEPTION_FEATURE_COLS,
    discover_task_files,
    list_subject_archives,
    load_e4_signal,
    load_emotion_signal,
    resample_to_grid,
    subject_sort_key,
)


TARGET_HZ = 4.0
WINDOW_SEC = 20.0
STEP_SEC = 5.0
WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_HZ)
STEP_SAMPLES = int(STEP_SEC * TARGET_HZ)


def finite_or_zero(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def peak_summary(values: np.ndarray) -> tuple[int, float, float]:
    if len(values) < 3 or not np.isfinite(values).any():
        return 0, 0.0, 0.0
    clean = pd.Series(values).interpolate(limit_direction="both").to_numpy(dtype=float)
    peaks, props = find_peaks(clean, width=5, prominence=0)
    amplitude = float(np.sum(props.get("prominences", np.array([]))))
    duration = float(np.sum(props.get("widths", np.array([]))) / TARGET_HZ)
    return int(len(peaks)), amplitude, duration


def landmark_distance(points: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a] - points[b]))


def au_values_from_landmarks(points: np.ndarray) -> list[float]:
    face_width = max(landmark_distance(points, 0, 16), 1.0)
    mouth_width = landmark_distance(points, 48, 54) / face_width
    mouth_height = landmark_distance(points, 51, 57) / face_width
    inner_mouth_height = landmark_distance(points, 62, 66) / face_width
    left_eye = (landmark_distance(points, 37, 41) + landmark_distance(points, 38, 40)) / (2.0 * face_width)
    right_eye = (landmark_distance(points, 43, 47) + landmark_distance(points, 44, 46)) / (2.0 * face_width)
    brow_eye = ((points[37, 1] - points[19, 1]) + (points[44, 1] - points[24, 1])) / (2.0 * face_width)
    cheek = ((points[48, 1] - points[31, 1]) + (points[54, 1] - points[35, 1])) / (2.0 * face_width)
    return [
        -mouth_width,
        (points[57, 1] - points[33, 1]) / face_width,
        landmark_distance(points, 51, 33) / face_width,
        mouth_width,
        -inner_mouth_height if np.isfinite(inner_mouth_height) else -mouth_height,
        (left_eye + right_eye + brow_eye) / 2.0,
        -cheek,
    ]


def load_landmark_signal(archive: Path, member: str) -> pd.DataFrame:
    import zipfile

    with zipfile.ZipFile(archive) as zf:
        obj = json.loads(zf.read(member))
    rows = []
    times = []
    keys = sorted((k for k in obj if k != "frame" and str(k).isdigit()), key=lambda value: int(value))
    for key in keys:
        frame = obj[key]
        try:
            points = np.array([[float(frame[str(i)]["x"]), float(frame[str(i)]["y"])] for i in range(68)], dtype=float)
        except Exception:
            continue
        if not np.isfinite(points).all():
            continue
        rows.append(au_values_from_landmarks(points))
        times.append(int(key) / 30.0)
    if not rows:
        raise ValueError(f"{member} has no usable landmark frames")
    return pd.DataFrame(rows, columns=AU_COLS).assign(time_rel=times)


def compute_window_features(win: pd.DataFrame, face_cols: list[str], face_feature_cols: list[str], feature_cols: list[str]) -> dict[str, float]:
    features: dict[str, float] = {}
    for col, feature_col in zip(face_cols, face_feature_cols):
        features[feature_col] = finite_or_zero(np.nanmean(win[col].to_numpy(dtype=float)))

    for signal in ["eda", "hr", "temp"]:
        arr = win[signal].to_numpy(dtype=float)
        arr = pd.Series(arr).interpolate(limit_direction="both").to_numpy(dtype=float)
        features[f"{signal}_mean"] = finite_or_zero(np.nanmean(arr))
        features[f"{signal}_min"] = finite_or_zero(np.nanmin(arr))
        features[f"{signal}_max"] = finite_or_zero(np.nanmax(arr))
        features[f"{signal}_std"] = finite_or_zero(np.nanstd(arr))

    eda = pd.Series(win["eda"].to_numpy(dtype=float)).interpolate(limit_direction="both").to_numpy(dtype=float)
    features["eda_kurtosis"] = finite_or_zero(kurtosis(eda, fisher=True, bias=False))
    features["eda_skew"] = finite_or_zero(skew(eda, bias=False))
    n_peaks, amp, dur = peak_summary(eda)
    features["eda_num_peaks"] = n_peaks
    features["eda_peak_amplitude"] = amp
    features["eda_peak_duration"] = dur

    hr = pd.Series(win["hr"].to_numpy(dtype=float)).interpolate(limit_direction="both").to_numpy(dtype=float)
    features["hr_rms"] = finite_or_zero(np.sqrt(np.nanmean(np.square(hr))))
    n_peaks, amp, dur = peak_summary(hr)
    features["hr_num_peaks"] = n_peaks
    features["hr_peak_amplitude"] = amp
    features["hr_peak_duration"] = dur
    return {col: features.get(col, 0.0) for col in feature_cols}


def face_schema(face_source: str) -> tuple[list[str], list[str], list[str]]:
    if face_source == "landmark":
        return AU_COLS, AU_FEATURE_COLS, AU_ALL_FEATURE_COLS
    return EMOTION_COLS, XCEPTION_FEATURE_COLS, FEATURE_COLS


def build_resampled_task(archive: Path, files, alignment: str = "absolute", face_source: str = "xception") -> pd.DataFrame | None:
    required_face = files.landmarks_path if face_source == "landmark" else files.xception_path
    if not all([required_face, files.hr_path, files.eda_path, files.temp_path]):
        return None
    if face_source == "landmark":
        face = load_landmark_signal(archive, files.landmarks_path)
        face_cols = AU_COLS
    else:
        face = load_emotion_signal(archive, files.xception_path)
        face_cols = EMOTION_COLS
    hr, hr_start, _ = load_e4_signal(archive, files.hr_path, "hr")
    eda, eda_start, _ = load_e4_signal(archive, files.eda_path, "eda")
    temp, temp_start, _ = load_e4_signal(archive, files.temp_path, "temp")

    task_start = min(hr_start, eda_start, temp_start)
    bio_frames = []
    max_times = [float(face["time_rel"].max())]
    for frame, col in [(hr, "hr"), (eda, "eda"), (temp, "temp")]:
        if alignment == "relative":
            rel = np.arange(len(frame), dtype=float) / TARGET_HZ if col in {"eda", "temp"} else np.arange(len(frame), dtype=float)
        else:
            rel = frame["time_abs"].to_numpy(dtype=float) - task_start
        frame = frame.copy()
        frame["time_rel"] = rel
        bio_frames.append((frame, col))
        if len(rel):
            max_times.append(float(np.nanmax(rel)))

    duration = min(max_times)
    if not np.isfinite(duration) or duration < WINDOW_SEC:
        return None
    if files.task_id != "T3" and duration > 540.0:
        duration = 540.0

    grid = np.arange(0.0, duration, 1.0 / TARGET_HZ)
    if len(grid) < WINDOW_SAMPLES:
        return None

    out = pd.DataFrame({"time_rel": grid})
    for col in face_cols:
        out[col] = resample_to_grid(
            face["time_rel"].to_numpy(dtype=float),
            face[col].to_numpy(dtype=float),
            grid,
        )
    for frame, col in bio_frames:
        out[col] = resample_to_grid(
            frame["time_rel"].to_numpy(dtype=float),
            frame[col].to_numpy(dtype=float),
            grid,
        )
    out["subject_id"] = files.subject_id
    out["task_id"] = files.task_id
    return out


def make_windows(data_root: Path, output_dir: Path, alignment: str = "absolute", face_source: str = "xception") -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    sequence_rows = []
    skipped = []
    face_cols, face_feature_cols, feature_cols = face_schema(face_source)
    sequence_cols = face_cols + ["eda", "hr", "temp"]

    for archive in list_subject_archives(data_root):
        task_map = discover_task_files(archive)
        for task_id in FEATURE_TASKS:
            files = task_map[task_id]
            try:
                resampled = build_resampled_task(archive, files, alignment=alignment, face_source=face_source)
            except Exception as exc:
                skipped.append({"subject_id": files.subject_id, "task_id": task_id, "reason": str(exc)})
                continue
            if resampled is None:
                skipped.append({"subject_id": files.subject_id, "task_id": task_id, "reason": "missing model-ready signals"})
                continue
            values = resampled[sequence_cols].to_numpy(dtype=np.float32)
            total = len(resampled)
            local_count = 0
            for start in range(0, total - WINDOW_SAMPLES + 1, STEP_SAMPLES):
                end = start + WINDOW_SAMPLES
                win = resampled.iloc[start:end]
                window_id = f"{files.subject_id}_{task_id}_{local_count:04d}"
                features = compute_window_features(win, face_cols, face_feature_cols, feature_cols)
                rows.append(
                    {
                        "window_id": window_id,
                        "subject_id": files.subject_id,
                        "task_id": task_id,
                        "site": files.site,
                        "start_sec": float(win["time_rel"].iloc[0]),
                        "end_sec": float(win["time_rel"].iloc[-1] + 1.0 / TARGET_HZ),
                        "archive_path": str(archive),
                        "xception_path": files.xception_path,
                        "hr_path": files.hr_path,
                        "eda_path": files.eda_path,
                        "temp_path": files.temp_path,
                        "alignment": alignment,
                        "face_source": face_source,
                        **features,
                    }
                )
                sequence_rows.append(values[start:end])
                local_count += 1

    features_df = pd.DataFrame(rows)
    if not features_df.empty:
        features_df = features_df.sort_values(
            ["subject_id", "task_id", "start_sec"],
            key=lambda s: s.map(subject_sort_key) if s.name == "subject_id" else s,
        ).reset_index(drop=True)
        order = {wid: i for i, wid in enumerate([row["window_id"] for row in rows])}
        sequence_order = [order[wid] for wid in features_df["window_id"]]
        sequences = np.stack(sequence_rows).astype(np.float32)[sequence_order]
    else:
        sequences = np.empty((0, WINDOW_SAMPLES, len(sequence_cols)), dtype=np.float32)

    features_path = output_dir / "windows_features.csv"
    metadata_path = output_dir / "windows_metadata.csv"
    sequence_path = output_dir / "windows_sequences.npz"
    features_df.to_csv(features_path, index=False)
    features_df[["window_id", "subject_id", "task_id", "site", "start_sec", "end_sec"]].to_csv(metadata_path, index=False)
    np.savez_compressed(sequence_path, window_id=features_df["window_id"].to_numpy(dtype=str), sequences=sequences, columns=np.array(sequence_cols))

    skipped_df = pd.DataFrame(skipped)
    skipped_path = output_dir / "windows_skipped.csv"
    skipped_df.to_csv(skipped_path, index=False)
    return features_df, skipped_df, sequences
