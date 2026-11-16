# EmpathicSchool supplementary analysis

The code recreates the direct-rating and hybrid-target analyses in Supplementary Tables S3 and S4. Install the Python dependencies in the repository’s `requirements.txt` before running these commands from the repository root.

## Inputs

Request the [EmpathicSchool release](https://doi.org/10.5281/zenodo.15556502) from its provider. In the commands below, `ARCHIVES` is the local directory of authorized participant ZIP files. The feature extraction command writes these files in `runs/empathicschool-features`:

- `physio_only_features.csv` and `physio_only_sequences.npz`
- `multimodal_features.csv` and `multimodal_sequences.npz`

Keep the original input structure and participant codes in your local files. The code uses these codes for leave-one-participant-out splits, joins between questionnaires and windows, and the original fold seeds. The analysis scripts avoid distributing input archives, intermediate files and participant-level outputs.

## Cohort construction and evaluation

```text
python -B empathicschool/preprocessing/extract_features.py --archive-dir "ARCHIVES" --out-dir runs/empathicschool-features
python -B empathicschool/preprocessing/prepare_common_scale.py --archive-dir "ARCHIVES" --feature-dir runs/empathicschool-features --out-dir runs/empathicschool-inputs
python -B empathicschool/analysis/run_common_scale.py --prepared-dir runs/empathicschool-inputs/prepared --out-dir runs/empathicschool-results
```

The feature extraction command builds wearable and facial windows, checks for duplicate recordings, and writes the input matrices locally. See the [preprocessing instructions](preprocessing/README.md) for details. If you already have these locally generated feature and sequence files, start with the preparation command and point `--feature-dir` to their directory.

The preparation command reads the original interval ratings locally and builds the direct-rating and hybrid targets. Directly recorded integer responses use the common 0–20 scale assumption. Dividing by 10 and applying the two thresholds assigns original scores of 0–6, 7–13 and 14–20 to classes 0, 1 and 2, respectively. Missing direct responses are excluded. The hybrid target adds protocol-assigned zeros for the designated reference tasks. Analysis B drops the four participant–task records whose class assignments were sensitive to the scale assumption.

The final command performs the participant-held-out fits and evaluates the average window-level class probabilities at the original interval level. It writes `results_summary.csv` for S3 and `paired_A_vs_B_differences.csv` for S4. The paired bootstrap resamples the same participants in A and B and retains each analysis’s eligible intervals. Both reported CSVs contain aggregate results.

| Cohort | Participants | Intervals A → B | Windows A → B |
| --- | ---: | ---: | ---: |
| Wearable, direct ratings | 20 | 382 → 362 | 7,913 → 7,493 |
| Wearable, hybrid targets | 20 | 781 → 761 | 16,133 → 15,713 |
| Multimodal, direct ratings | 11 | 196 → 181 | 4,010 → 3,698 |
| Multimodal, hybrid targets | 11 | 400 → 385 | 8,211 → 7,899 |
