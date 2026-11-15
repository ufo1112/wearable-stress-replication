# Reproduce the two StressID manuscript figures

The plot script reads retained aggregate estimates and confidence intervals. It needs Python 3.10 or later, `reportlab`, and the DejaVu Sans regular and bold TrueType fonts.

```text
python figures/plot_stressid.py --context-csv PATH/TO/context_incremental_results.csv --transfer-csv PATH/TO/subject_task_disjoint_all_models.csv --out-dir reproduced_figures
```

If the fonts are not found, give `--font-dir PATH/TO/FONTS` with `DejaVuSans.ttf` and `DejaVuSans-Bold.ttf`. The script also looks in the Windows fonts directory, the common Linux DejaVu directory, and Matplotlib's bundled fonts when Matplotlib is installed.

- `Figure_1.pdf` uses the paired differences in Spearman correlation for combined versus context-only models, separately for continuous stress and arousal.
- `Figure_2.pdf` uses balanced accuracy for task-condition classification in the participant-held-out and joint participant-and-task-held-out designs.

Both figures are single-page vector PDFs. The script uses full-precision values from the CSV files for plotting. The confidence intervals are the retained participant-cluster bootstrap intervals.
