"""Reproduce the WESAD six-item STAI and questionnaire-to-window alignment."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


CONDITIONS = ("baseline", "stress", "amusement", "meditation")
QUESTIONNAIRE_CONDITIONS = {
    "Base": "baseline",
    "TSST": "stress",
    "Fun": "amusement",
    "Medi 1": "meditation",
    "Medi 2": "meditation",
}
LABEL_CONDITIONS = dict(enumerate(CONDITIONS, start=1))
SUBJECTS = tuple(f"S{number}" for number in range(2, 18) if number != 12)
LABEL_HZ = 700
GRID_HZ = 4
BVP_HZ = 64
ACC_HZ = 32
WINDOW_SAMPLES = 20 * GRID_HZ
STEP_SAMPLES = 5 * GRID_HZ


def six_numeric_values(fields):
    values = []
    for field in fields:
        try:
            values.append(float(field.strip().replace(",", ".")))
        except ValueError:
            continue
    if len(values) < 6 or any(value not in {1, 2, 3, 4} for value in values[:6]):
        raise ValueError("Expected six STAI responses scored from 1 to 4")
    return values[:6]


def condition_scores(path: Path) -> tuple[dict[str, float], int]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    order_line = next(line for line in lines if line.startswith("# ORDER"))
    order = [field.strip() for field in order_line.split(";")[1:]]
    order = [field for field in order if field in QUESTIONNAIRE_CONDITIONS][:5]
    stai_lines = [line for line in lines if line.startswith("# STAI")]
    if set(order) != set(QUESTIONNAIRE_CONDITIONS) or len(stai_lines) != 5:
        raise ValueError(f"{path.name}: expected five STAI administrations")

    scores = defaultdict(list)
    for condition, line in zip(order, stai_lines):
        items = np.asarray(six_numeric_values(line.split(";")[1:]), dtype=float)
        # At ease, relaxed, and pleasant are items 1, 4, and 6.
        items[[0, 3, 5]] = 5 - items[[0, 3, 5]]
        scores[QUESTIONNAIRE_CONDITIONS[condition]].append(float(items.sum()))
    return {condition: float(np.mean(totals)) for condition, totals in scores.items()}, len(stai_lines)


def labels_on_wrist_grid(labels: np.ndarray, length: int) -> np.ndarray:
    grid = np.zeros(length, dtype=np.int16)
    transitions = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.r_[0, transitions]
    ends = np.r_[transitions, len(labels)]
    for start, end in zip(starts, ends):
        label = int(labels[start])
        if label not in LABEL_CONDITIONS:
            continue
        # Keep points at least one second from each edge of the protocol block.
        first = max(0, int(np.ceil((start + LABEL_HZ) / LABEL_HZ * GRID_HZ)))
        stop = min(length, int(np.floor((end - LABEL_HZ) / LABEL_HZ * GRID_HZ)))
        if stop > first:
            grid[first:stop] = label
    return grid


def iter_windows(path: Path):
    # Only unpickle data obtained directly from the official WESAD provider.
    with path.open("rb") as stream:
        data = pickle.load(stream, encoding="latin1")
    labels = data["label"]
    wrist = data["signal"]["wrist"]
    acceleration = np.asarray(wrist["ACC"])
    if acceleration.ndim != 2 or acceleration.shape[1] != 3:
        raise ValueError(f"{path.name}: expected three wrist acceleration channels")
    length = min(
        np.size(wrist["EDA"]),
        np.size(wrist["TEMP"]),
        np.size(wrist["BVP"]) // (BVP_HZ // GRID_HZ),
        len(acceleration) // (ACC_HZ // GRID_HZ),
        len(labels) // (LABEL_HZ // GRID_HZ),
    )
    grid = labels_on_wrist_grid(labels, length)
    for start in range(0, length - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        window_labels = grid[start : start + WINDOW_SAMPLES]
        if np.all(window_labels):
            # The original extraction used the most frequent label, lowest ID on ties.
            label = int(np.bincount(window_labels).argmax())
            yield f"{path.stem}_w{start:06d}", LABEL_CONDITIONS[label]


def summarize(data_root: Path, retained_windows: Path | None) -> dict:
    scores_by_condition = defaultdict(list)
    window_scores = defaultdict(set)
    expected_window_rows = {}
    administration_count = 0
    stress_strictly_highest = 0
    window_count = 0
    for subject in SUBJECTS:
        subject_dir = data_root / subject
        scores, administrations = condition_scores(subject_dir / f"{subject}_quest.csv")
        administration_count += administrations
        for condition in CONDITIONS:
            scores_by_condition[condition].append(scores[condition])
        stress_strictly_highest += scores["stress"] > max(
            scores[condition] for condition in CONDITIONS if condition != "stress"
        )
        for window_id, condition in iter_windows(subject_dir / f"{subject}.pkl"):
            window_count += 1
            score = scores[condition]
            window_scores[subject, condition].add(score)
            if retained_windows is not None:
                expected_window_rows[window_id] = (subject, condition, score)

    summary = {
        "participants": len(SUBJECTS),
        "stai_administrations": administration_count,
        "participant_condition_scores": sum(map(len, scores_by_condition.values())),
        "windows": window_count,
        "participant_condition_window_groups": len(window_scores),
        "groups_with_within_condition_stai_variation": sum(
            len(scores) > 1 for scores in window_scores.values()
        ),
        "participants_with_stress_strictly_highest": stress_strictly_highest,
        "condition_mean_stai6": {
            condition: round(float(np.mean(scores_by_condition[condition])), 4)
            for condition in CONDITIONS
        },
    }
    if retained_windows is not None:
        with retained_windows.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                expected = expected_window_rows.pop(row["window_id"], None)
                observed = (row["subject_id"], row["condition"], float(row["stai"]))
                if observed != expected:
                    raise SystemExit("Reconstructed windows differ from the retained alignment file.")
        if expected_window_rows:
            raise SystemExit("The retained alignment file has fewer rows than the reconstruction.")
        summary["retained_window_comparison"] = "all rows match"
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Extracted official WESAD directory")
    parser.add_argument(
        "--verify-retained-windows",
        type=Path,
        help="Optional local CSV of the original window ID, subject ID, condition, and STAI scores",
    )
    args = parser.parse_args()
    print(json.dumps(summarize(args.data_root, args.verify_retained_windows), indent=2))


if __name__ == "__main__":
    main()
