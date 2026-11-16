# Reproduce EmpathicSchool feature extraction

Obtain the original EmpathicSchool participant ZIPs through the [dataset record](https://doi.org/10.5281/zenodo.15556502) and the original providers' access conditions. Keep the archives intact in one local directory. This repository contains source code only. Store feature matrices, model inputs, and intermediate outputs in a separate private directory.

This workflow uses Python with NumPy, pandas, SciPy, NeuroKit2, openpyxl and PyArrow. See the root environment requirements for versions.

```text
python empathicschool/preprocessing/extract_features.py --archive-dir AUTHORIZED_ZIPS --out-dir PRIVATE_FEATURES
python empathicschool/preprocessing/prepare_common_scale.py --archive-dir AUTHORIZED_ZIPS --feature-dir PRIVATE_FEATURES --out-dir PRIVATE_ANALYSIS
```

Use new output directories. The first command reconstructs the landmark-window features using absolute-time alignment, flags the prior questionnaire eligibility cohort, detects duplicate feature sets, extracts the 17 wearable features and six wearable sequence channels, and joins the seven landmark features for the multimodal input. It uses the supplied landmark JSON files and does not process raw video.

The second command builds the eight Analysis A/B tables and two aligned sequence arrays under `PRIVATE_ANALYSIS/prepared`. It reads the original five interval stress responses, applies the common-scale assumption, handles missing responses without carrying values forward, and keeps protocol zeros in the hybrid target. The four Analysis B exclusions are recovered by comparing the original-slot classifications with the historical rule. The preparation command also accepts authorized, already-extracted feature pairs using `--feature-dir`, and does not need the historical prediction files.

The retained feature directory contains `physio_only_features.csv`, `physio_only_sequences.npz`, `multimodal_features.csv` and `multimodal_sequences.npz`. On the archived files, preparation reproduced the analysis tables and sequence arrays exactly.
