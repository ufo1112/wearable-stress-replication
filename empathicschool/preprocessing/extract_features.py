"""Create both EmpathicSchool modality inputs from authorized dataset archives."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from duplicate_check import detect_duplicate_subjects
from empathic_common import (AU_FEATURE_COLS, FEATURE_TASKS, discover_task_files,
                            list_subject_archives, load_label_sheet, stress_intervals_for_task,
                            subject_sort_key)
from facial_windows import make_windows
from legacy_facial_labels import make_labels
from wearable_features import (MULTIMODAL_SEQUENCE_COLS, PHYSIO_SEQUENCE_COLS, STEP_SAMPLES,
                               TARGET_HZ, WINDOW_SAMPLES, label_values, load_task_timeseries,
                               window_features)


def eligibility_exclusions(archives):
    """Recover the archived self-report eligibility set from available files."""
    rating_tasks = ("T2", "T3", "T5", "T7")
    excluded = set()
    for archive in list_subject_archives(archives):
        tasks = discover_task_files(archive)
        subject = tasks[rating_tasks[0]].subject_id
        all_streams = all(tasks[task].eda_path and tasks[task].temp_path and tasks[task].acc_path
                          for task in rating_tasks)
        labels = load_label_sheet(archive, tasks[rating_tasks[0]].label_path) if all_streams else None
        has_rating = any(stress_intervals_for_task(labels, task) is not None for task in rating_tasks)
        if not (all_streams and has_rating):
            excluded.add(subject)
    return excluded


def extract_wearable(archive_dir, removed, excluded_subjects):
    rows, sequences = [], []
    for archive in list_subject_archives(archive_dir):
        subject = f"S{subject_sort_key(archive)}"
        if subject in removed:
            continue
        task_map = discover_task_files(archive)
        try:
            labels = load_label_sheet(archive, next(iter(task_map.values())).label_path)
        except Exception as exc:
            logging.warning("Questionnaire unavailable for %s: %s", subject, exc)
            labels = None
        for task in FEATURE_TASKS:
            files = task_map[task]
            series, status = load_task_timeseries(archive, files)
            if series is None:
                continue
            values = series[PHYSIO_SEQUENCE_COLS].to_numpy(dtype=np.float32)
            starts = range(0, len(series) - WINDOW_SAMPLES + 1, STEP_SAMPLES)
            for local_index, start in enumerate(starts):
                end = start + WINDOW_SAMPLES
                window = series.iloc[start:end]
                start_sec = float(window.time_rel.iloc[0])
                end_sec = float(window.time_rel.iloc[-1] + 1.0 / TARGET_HZ)
                rows.append({
                    "window_id": f"{subject}_{task}_{local_index:04d}", "subject_id": subject,
                    "task_id": task, "site": files.site, "start_sec": start_sec, "end_sec": end_sec,
                    "archive_path": str(archive), "eda_path": files.eda_path, "temp_path": files.temp_path,
                    "acc_path": files.acc_path, "hr_path": files.hr_path, "bvp_path": files.bvp_path,
                    "ibi_path": files.ibi_path, "hrv_source": status["hrv_source"],
                    "eda_decomposition": status["eda_method"], **window_features(window),
                    **label_values(labels, subject, task, start_sec, end_sec, excluded_subjects),
                })
                sequences.append(values[start:end])
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No complete wearable windows were extracted")
    order = frame.sort_values(["subject_id", "task_id", "start_sec"],
                              key=lambda col: col.map(subject_sort_key) if col.name == "subject_id" else col).index
    return frame.loc[order].reset_index(drop=True), np.stack(sequences).astype(np.float32)[order]


def join_face_features(wearable, wearable_sequences, face):
    joined = wearable.merge(face[["window_id"] + AU_FEATURE_COLS], on="window_id", how="inner")
    index = pd.Index(wearable.window_id).get_indexer(joined.window_id)
    base = wearable_sequences[index]
    face_values = joined[AU_FEATURE_COLS].to_numpy(dtype=np.float32)
    face_sequences = np.repeat(face_values[:, None, :], base.shape[1], axis=1)
    return joined, np.concatenate([base, face_sequences], axis=2).astype(np.float32)


def save_modality(directory, name, frame, sequences, channels):
    frame.to_csv(directory / f"{name}_features.csv", index=False)
    np.savez_compressed(directory / f"{name}_sequences.npz", sequences=sequences,
                        window_id=frame.window_id.to_numpy(dtype=str), columns=np.asarray(channels))
    print(f"{name}: {len(frame)} windows, {frame.subject_id.nunique()} participants", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="New private feature output directory")
    args = parser.parse_args()
    archives = args.archive_dir.resolve()
    output = args.out_dir.resolve()
    if not list_subject_archives(archives):
        parser.error("--archive-dir must contain the authorized participant ZIP files")
    exclusions = eligibility_exclusions(archives)
    output.mkdir(parents=True, exist_ok=False)
    face_outputs = output / "facial_intermediate"
    print("Extracting landmark windows", flush=True)
    make_windows(archives, face_outputs, alignment="absolute", face_source="landmark")
    labeled_path = face_outputs / "windows_labeled.csv"
    make_labels(face_outputs / "windows_features.csv", labeled_path, exclusions)
    labeled = pd.read_csv(labeled_path)
    _, removed = detect_duplicate_subjects(labeled)
    face = labeled[~labeled.subject_id.isin(removed)]
    dedup_path = face_outputs / "windows_labeled_dedup.csv"
    face.to_csv(dedup_path, index=False)
    face = pd.read_csv(dedup_path)
    print(f"Extracting wearable windows after excluding {len(removed)} duplicate participant archive", flush=True)
    wearable, sequences = extract_wearable(archives, removed, exclusions)
    save_modality(output, "physio_only", wearable, sequences, PHYSIO_SEQUENCE_COLS)
    combined, combined_sequences = join_face_features(wearable, sequences, face)
    save_modality(output, "multimodal", combined, combined_sequences, MULTIMODAL_SEQUENCE_COLS)


if __name__ == "__main__":
    main()
