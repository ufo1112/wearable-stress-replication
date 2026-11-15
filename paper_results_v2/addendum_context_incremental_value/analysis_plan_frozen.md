# Frozen Analysis Plan: StressID Context and Incremental Value

Frozen before inspecting retained outcome values on 2026-07-25.

## Status and Scope

- Branch: `paper_revision_measurement_units`
- Primary dataset: StressID
- Primary independent unit: participant-task rating event
- Outer validation: strict leave-one-subject-out (LOSO)
- Primary analyses: context/physiology/combined comparisons and within-task residual prediction
- Post-hoc explanatory analyses: variance decomposition and subject-and-task-disjoint transfer
- Diagnostic-only analyses: uncertainty comparison by resampling unit
- No model, feature, target, aggregation, or reporting choice will be changed after outcome inspection.

## Fixed Inputs and Leakage Rules

- Reuse the retained StressID feature matrix, participant/task keys, labels, and fixed RF, XGBoost, and LightGBM settings from `paper_results_v2`.
- Reuse the same subject membership and outer LOSO folds.
- All imputation, scaling, feature encoding, and model fitting use outer-training participants only.
- Task means and binary task prevalences are estimated from outer-training participants only.
- No held-out participant rating is used to define context predictions, residual targets, task means, preprocessing, or model selection.
- Window predictions are aggregated to participant-task events by arithmetic mean probability or arithmetic mean continuous prediction.
- Each participant-task event receives equal training weight through inverse window count.
- Random seed: 123.
- Bootstrap resamples: 2,000.
- Bootstrap confidence interval: percentile 2.5th and 97.5th percentiles.

## A. Context and Incremental Value

Targets:

1. Continuous post-task stress rating.
2. Continuous post-task arousal rating.
3. Dataset-provided binary stress label.

Predictor sets:

1. Context-only: outer-training mean rating or positive-label prevalence for the exact task identity.
2. Protocol-block/order-only: use a distinct documented block/order variable if one exists. If no distinct variable exists, report this baseline as unavailable rather than constructing one after seeing outcomes.
3. Physiology-only: current engineered physiological features with fixed RandomForest, XGBoost, and LightGBM settings.
4. Context + physiology: append training-fitted one-hot task identity to the same physiological matrix.

Task identity is a diagnostic context baseline, not a deployment feature.

Continuous metrics:

- Pooled Spearman rho, MAE, and RMSE over participant-task events.
- Subject-cluster bootstrap confidence intervals.

Binary metrics:

- Pooled balanced accuracy and macro-F1 over participant-task events.
- Subject-cluster bootstrap confidence intervals.

Paired differences use exactly matched participant-task events:

- Combined minus context-only.
- Physiology-only minus context-only.
- Combined minus physiology-only.

Differences are always reported literally as left minus right. Thus positive is favorable for Spearman, balanced accuracy, and macro-F1; negative is favorable for MAE and RMSE.

## B. Within-Task Residual Prediction

For each outer fold and target:

1. Estimate each task mean from outer-training participants only.
2. Define the held-out residual as observed rating minus that training-only task mean.
3. Fit each fixed physiology model to training residuals defined using means calculated within that same outer-training set.
4. Compare with a fixed zero-residual predictor.

Report pooled Spearman rho, MAE, RMSE, and R-squared when the pooled target variance is nonzero, with participant-cluster bootstrap confidence intervals. Per-task metrics are supplementary and use no additional tuning.

## C. Variance Decomposition

For stress and arousal, fit a descriptive repeated-measures random-intercept model:

`rating ~ C(task) + (1 | participant)`

Report participant random-intercept variance, residual variance, marginal R-squared for fixed task/protocol context, conditional R-squared including participant, model convergence, and task-level distributions. Interpret task identity only as task/protocol-context because task and order are confounded.

## D. Uncertainty by Analysis Unit

Use identical retained OOF predictions for:

1. Task-condition balanced accuracy.
2. Dataset binary-stress balanced accuracy.
3. Paired task-minus-stress balanced-accuracy difference on common events.

Compare:

- Naive window bootstrap: diagnostic only and invalid for primary inference.
- Participant-task cluster bootstrap.
- Participant cluster bootstrap: primary inference.

Use 2,000 resamples for each method and report point estimate, interval endpoints and width, windows, events, and participants. The paired contrast resamples matched common records/clusters and recomputes both metrics in every draw.

## E. Subject-and-Task-Disjoint Transfer

This post-hoc analysis is attempted only if the required training design is computationally feasible without changing fixed settings. For each held-out participant-task event, exclude all windows from both the held-out participant and held-out task identity, fit the primary RandomForest, aggregate the held-out event prediction, and pool predictions for balanced accuracy with participant-cluster confidence intervals. If not run, report the measured feasibility reason and no result.

## Reporting Commitments

- Report every requested model/target/metric outcome, including failures and undefined metrics.
- Do not describe nonsignificant results as being at chance.
- Do not select models or outcomes by performance.
- Clearly label primary, post-hoc explanatory, and diagnostic-only outputs.
- Write all new outputs only under `paper_results_v2/addendum_context_incremental_value/`.
