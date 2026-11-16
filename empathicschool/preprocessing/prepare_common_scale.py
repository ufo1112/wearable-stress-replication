"""Recreate the common-scale Analysis A and B inputs from authorized local files."""

import argparse
import io
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import openpyxl
import pandas as pd

from empathic_common import row_for_task, stress_intervals_for_task


OBSERVED = ("T2", "T3", "T5", "T7")
PROTOCOL = ("T1", "T4", "T6", "T8")
BOUNDARIES = (90, 210, 330, 450, np.inf)
RATING_HEADERS = (
    ("Stress_first_interval",), ("Stress_second_interval",), ("Stress_third_interval",),
    ("Stress_forth_interval_", "Stress_fourth_interval"), ("Stress_fifth_interval",),
)
PHYSIO_FEATURES = (
    "eda_scr_peak_count", "eda_scr_amp_mean", "eda_scr_amp_max", "eda_scr_rise_time_mean",
    "eda_phasic_auc", "eda_slope", "eda_tonic_mean", "hrv_mean_ibi", "hrv_rmssd", "hrv_sdnn",
    "hrv_pnn50", "temp_mean", "temp_slope", "temp_max_drop", "acc_mag_mean", "acc_mag_std",
    "acc_mag_max",
)
PHYSIO_CHANNELS = ("eda", "eda_phasic", "eda_tonic", "ibi_ms", "temp", "acc_mag")
FACE_FEATURES = (
    "lip_puckerer_mean", "lower_lip_depressor_mean", "upper_lip_raiser_mean", "mouth_stretch_mean",
    "lip_tightener_mean", "upper_lid_raiser_mean", "cheek_raiser_mean",
)
MODALITIES = {
    "physio_only": (PHYSIO_FEATURES, PHYSIO_CHANNELS),
    "multimodal": (PHYSIO_FEATURES + FACE_FEATURES, PHYSIO_CHANNELS + FACE_FEATURES),
}
METADATA_COLUMNS = (
    "window_id", "subject_id", "participant_id", "task_id", "interval_index", "event_id",
    "interval_id", "start_sec", "end_sec", "raw_value", "rating_value", "y_true", "label_source",
    "is_imputed", "variant", "dataset_variant", "mode", "target", "_sequence_index",
)
OBSERVED_COUNTS = {
    ("physio_only", "A"): (20, 382, 7913), ("physio_only", "B"): (20, 362, 7493),
    ("multimodal", "A"): (11, 196, 4010), ("multimodal", "B"): (11, 181, 3698),
}


def workbook_bytes(archive):
    with ZipFile(archive) as bundle:
        candidates = [name for name in bundle.namelist()
                      if name.lower().endswith((".xlsx", ".xls"))
                      and "xception" not in name.lower() and not name.startswith("__MACOSX")
                      and not any(part.startswith(".") or re.fullmatch(r"[Tt][1-9]", part)
                                  for part in name.split("/"))]
        if len(candidates) != 1:
            raise ValueError(f"Expected one questionnaire workbook in {archive.name}")
        return bundle.read(candidates[0])


def category(value):
    if value <= 0.65:
        return 0
    return 1 if value <= 1.3 else 2


def load_ratings(subjects, archive_dir):
    ratings = {}
    sensitive = set()
    for subject in sorted(subjects):
        data = workbook_bytes(archive_dir / f"{subject}.zip")
        frame = pd.read_excel(io.BytesIO(data))
        book = openpyxl.load_workbook(io.BytesIO(data), data_only=False, read_only=True)
        try:
            sheet = book.worksheets[0]
            headers = [cell.value for cell in sheet[1]]
            interval_columns = []
            for choices in RATING_HEADERS:
                present = [name for name in choices if name in frame.columns]
                if len(present) != 1:
                    raise ValueError(f"Expected one workbook column from {choices}")
                interval_columns.append(headers.index(present[0]) + 1)
            for task in OBSERVED:
                row = row_for_task(frame, task)
                if row is None:
                    continue
                historical = stress_intervals_for_task(frame, task) or [np.nan] * 5
                for slot, col in enumerate(interval_columns):
                    value = sheet.cell(row=int(row.name) + 2, column=col).value
                    if str(value).strip().lower() in {"", "na", "n/a", "nan", "unknown", "none"}:
                        continue
                    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                    if pd.isna(numeric) or not np.isfinite(numeric) or not 0 <= numeric <= 20:
                        raise ValueError("A non-missing original rating falls outside the assumed 0–20 scale")
                    ratings[f"{subject}|{task}|I{slot}"] = float(numeric)
                    if not np.isfinite(historical[slot]) or category(historical[slot]) != category(numeric / 10):
                        sensitive.add((subject, task))
        finally:
            book.close()
    if len(sensitive) != 4:
        raise ValueError(f"Expected the four archived scale-sensitivity records, found {len(sensitive)}")
    return ratings, sensitive


def load_windows(feature_dir, modality):
    table = pd.read_csv(feature_dir / f"{modality}_features.csv", low_memory=False)
    table = table.drop_duplicates("window_id").drop_duplicates()
    flagged = table.is_labeled_analysis.astype(str).str.lower().isin({"true", "1", "yes"})
    eligible = set(table.loc[flagged, "subject_id"].astype(str))
    return table[table.subject_id.astype(str).isin(eligible)].reset_index(drop=True)


def prepare_modality(raw, modality, feature_dir, output, ratings, sensitive):
    features, channels = MODALITIES[modality]
    with np.load(feature_dir / f"{modality}_sequences.npz", allow_pickle=True) as sequences:
        ids = pd.Index(sequences["window_id"].astype(str))
        if not ids.is_unique or tuple(sequences["columns"]) != channels:
            raise ValueError(f"Unexpected sequence identifiers or channels for {modality}")
        lookup = ids.get_indexer(raw.window_id.astype(str))
        if (lookup < 0).any():
            raise ValueError(f"A {modality} window is missing its input sequence")
        values = sequences["sequences"][lookup].astype(np.float32)
    if values.shape[1:] != (80, len(channels)) or not np.allclose(raw.end_sec - raw.start_sec, 20):
        raise ValueError(f"Expected complete 20-second windows with 4-Hz sequences for {modality}")
    np.savez_compressed(output / f"{modality}_sequences.npz", sequences=values,
                        window_id=raw.window_id.astype(str).to_numpy(), columns=np.asarray(channels))

    midpoint = (raw.start_sec + raw.end_sec) / 2
    raw["interval_index"] = np.searchsorted(BOUNDARIES, midpoint, side="right").clip(0, 4)
    raw["event_id"] = raw.subject_id + "|" + raw.task_id + "|I" + raw.interval_index.astype(str)
    raw["_sequence_index"] = np.arange(len(raw))
    raw["raw_value"] = raw.event_id.map(ratings)
    direct = raw.task_id.isin(OBSERVED) & raw.raw_value.notna()
    protocol = raw.task_id.isin(PROTOCOL)

    for mode in ("observed", "hybrid"):
        selected = raw[direct | protocol] if mode == "hybrid" else raw[direct]
        selected = selected.copy()
        selected["label_source"] = np.where(selected.task_id.isin(PROTOCOL), "protocol", "self_report")
        selected["rating_value"] = np.where(selected.label_source.eq("protocol"), 0, selected.raw_value / 10)
        selected["y_true"] = selected.rating_value.map(category)
        selected["participant_id"] = selected.subject_id
        selected["interval_id"] = selected.event_id
        selected["is_imputed"] = False
        selected["target"] = "observed_self_report" if mode == "observed" else "hybrid_self_report_protocol"
        selected["dataset_variant"] = modality
        selected["mode"] = mode
        keep_for_b = [(sid, task) not in sensitive for sid, task in zip(selected.subject_id, selected.task_id)]
        for variant, subset in (("A", selected), ("B", selected.loc[keep_for_b])):
            subset = subset.assign(variant=variant)
            counts = (subset.subject_id.nunique(), subset.event_id.nunique(), len(subset))
            if mode == "observed" and counts != OBSERVED_COUNTS[modality, variant]:
                raise ValueError(f"{modality}/{variant} observed cohort differs from archived run02: {counts}")
            subset[list(METADATA_COLUMNS + features)].to_parquet(output / f"{modality}__{mode}__{variant}.parquet", index=False)
            print(f"{modality} {mode} {variant}: {counts[0]} participants, {counts[1]} intervals, {counts[2]} windows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, help="Local root containing morning3_run/ and 15556502/")
    parser.add_argument("--archive-dir", type=Path, help="Directory of authorized participant ZIP archives")
    parser.add_argument("--feature-dir", type=Path, help="Directory containing both modality feature/sequence pairs")
    parser.add_argument("--out-dir", type=Path, required=True, help="New local output directory for private inputs")
    args = parser.parse_args()
    archive_dir = args.archive_dir or (args.project_root / "15556502" if args.project_root else None)
    feature_dir = args.feature_dir or (args.project_root / "morning3_run" if args.project_root else None)
    if archive_dir is None or feature_dir is None:
        parser.error("supply --project-root or both --archive-dir and --feature-dir")
    windows = {modality: load_windows(feature_dir, modality) for modality in MODALITIES}
    subjects = set().union(*(set(frame.subject_id) for frame in windows.values()))
    ratings, sensitive = load_ratings(subjects, archive_dir)
    output = args.out_dir / "prepared"
    output.mkdir(parents=True, exist_ok=False)
    for modality, frame in windows.items():
        prepare_modality(frame, modality, feature_dir, output, ratings, sensitive)


if __name__ == "__main__":
    main()
