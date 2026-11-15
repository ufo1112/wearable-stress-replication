# WESAD questionnaire alignment

This analysis reproduces the WESAD questionnaire scores and counts reported in the manuscript. Each participant contributes one score per condition. Assigning that score to wrist-recording windows checks alignment and creates no additional self-reports.

## Data and execution

Obtain WESAD from the [University of Siegen](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) or its [UCI record](https://doi.org/10.24432/C57K5T). Use only participant pickle files from a trusted official release, because opening a pickle can execute code. Point `--data-root` to the directory with the 15 participant folders. The runner reads each participant's official questionnaire CSV and synchronized pickle locally. No data are included in this directory.

From the `wesad` directory, with Python 3.10 or later:

```sh
python -m pip install -r requirements.txt
python questionnaire_alignment.py --data-root /path/to/WESAD
```

The program prints only aggregate results. An optional `--verify-retained-windows` argument accepts a local CSV containing `window_id`, `subject_id`, `condition`, and `stai` to compare each reconstructed row against an existing analysis file. That CSV is not required to run the analysis and is not distributed.

## Scoring and alignment

The script reads five STAI administrations per participant in recorded condition order. It reverse-scores the anxiety-absent items (at ease, relaxed, and pleasant), sums the six responses, and averages the two meditation administrations. For alignment it reconstructs the original 4 Hz wrist grid, trims one second from each end of labeled blocks, and forms 20-second windows with a 5-second step. Each retained window contains only valid protocol labels and receives the participant–condition score for its majority label.

The retained release yields 15 participants, 75 administrations, 60 participant–condition scores and 8,665 windows. Mean STAI-6 scores are 10.80 for baseline, 18.53 for stress, 9.27 for amusement and 10.13 for meditation. None of the 60 participant–condition groups shows within-group variation after the window assignment.

The implementation uses the original window-selection logic from `wesad_extract.py`, the questionnaire-to-window merge in `wesad_selfreport.py`, and the corrected STAI scoring from `build_measurement_addendum.py`.
