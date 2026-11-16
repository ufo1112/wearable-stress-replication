# EmpathicSchool common-scale model evaluation

This runner refits the common-scale analyses for Supplementary Tables S3 and S4. It takes the private prepared inputs created by `../preprocessing/prepare_common_scale.py` and writes two aggregate CSVs.

Follow [the EmpathicSchool instructions](../README.md) to create the required local `prepared` directory. To run the evaluation on an existing prepared directory, use Python 3.10 from the replication package root:

```sh
python -m pip install -r empathicschool/analysis/requirements.txt
python empathicschool/analysis/run_common_scale.py --prepared-dir /path/to/new/prepared-run/prepared --out-dir /path/to/new/model-results
```

Keep the prepared tables and sequences private. Preserve the original local participant keys and row order to reproduce the participant-specific random seeds and model fits. The runner does not require the original development checkout, participant-level prediction files, or the earlier analysis outputs.

Each modality and target is evaluated under both the common 0–20 scale analysis (A) and the scale sensitivity analysis (B). The 32 model configurations use leave-one-subject-out evaluation with equal total training weight per interval. Class probabilities are averaged within each original interval. `results_summary.csv` reports the pooled, fixed three-class balanced accuracy and macro-F1 with 95% participant-cluster bootstrap intervals. `paired_A_vs_B_differences.csv` reports B minus A with the same 2,000 participant resamples applied to both analyses. Each analysis retains its own set of intervals. The bootstraps use seed 123 and reuse held-out predictions without refitting.

The model settings, transforms, CNN and seed function come from the archived common-scale run02 scripts and their frozen model helpers. The original run used PyTorch 2.5.1+cu121 on Windows and fitted the CNN on CPU. Other compatible builds may differ slightly in floating-point results. The runner emits progress by analysis group and saves the two aggregate tables after all folds complete. It saves no participant-level results.
