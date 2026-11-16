"""Archived duplicate check over sorted, rounded facial-window feature rows."""

import itertools
import json

import numpy as np
import pandas as pd

from empathic_common import AU_ALL_FEATURE_COLS as FEATURE_COLS, subject_sort_key

def feature_matrix_hash(df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS) -> str:
    arr = df[feature_cols].to_numpy(dtype=float)
    arr = np.round(arr, 8)
    order = np.lexsort(arr.T[::-1])
    return json.dumps(arr[order].tolist(), separators=(",", ":"))


def detect_duplicate_subjects(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    labeled = df[df["is_labeled_analysis"].astype(bool)].copy()
    hashes = {}
    for subject, group in labeled.groupby("subject_id"):
        hashes[subject] = feature_matrix_hash(group, FEATURE_COLS)
    pairs = []
    for a, b in itertools.combinations(sorted(hashes, key=subject_sort_key), 2):
        identical = hashes[a] == hashes[b]
        pairs.append(
            {
                "subject_a": a,
                "subject_b": b,
                "windows_a": int((labeled["subject_id"] == a).sum()),
                "windows_b": int((labeled["subject_id"] == b).sum()),
                "feature_multiset_identical": bool(identical),
            }
        )
    pair_df = pd.DataFrame(pairs)
    duplicate_edges = pair_df[pair_df["feature_multiset_identical"]]
    graph: dict[str, set[str]] = {}
    for row in duplicate_edges.itertuples(index=False):
        graph.setdefault(row.subject_a, set()).add(row.subject_b)
        graph.setdefault(row.subject_b, set()).add(row.subject_a)
    remove_subjects = []
    visited = set()
    for subject in sorted(graph, key=subject_sort_key):
        if subject in visited:
            continue
        stack = [subject]
        component = set()
        while stack:
            cur = stack.pop()
            if cur in component:
                continue
            component.add(cur)
            stack.extend(graph.get(cur, set()) - component)
        visited |= component
        keep = sorted(component, key=subject_sort_key)[0]
        remove_subjects.extend([s for s in sorted(component, key=subject_sort_key) if s != keep])
    return pair_df, remove_subjects
