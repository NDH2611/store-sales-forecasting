# TV4 Technical Report: Evaluation and Explainability

## Scope

This report consolidates the available rolling-origin validation evidence and the
Champion LightGBM SHAP analysis. Validation scores are reported on the sales
scale (RMSLE/MAE); Tree SHAP attributions are on the model's raw
`log1p(sales)` output scale. These scales are not interchangeable.

The project uses 16-day forecasts over 1,782 store-family series. The three
reported tuning origins are `2017-06-12`, `2017-06-28`, and `2017-07-14`, each
with 28,512 expected rows, for 85,536 validation rows in total.

## Evaluation evidence and Champion selection

The Champion is LightGBM configuration `opt_730_l63_mcs35_reg` (730 training
days, `holiday` feature set, 1,400 estimators, 63 leaves, minimum 35 samples
per leaf, row/column sampling 0.8/0.6, L1/L2 regularization 0.1/1.0; seed 42).
The logged fold scores are:

| Fold origin | RMSLE | MAE | Rows |
|---|---:|---:|---:|
| 2017-06-12 | 0.389230 | 62.847738 | 28,512 |
| 2017-06-28 | 0.380427 | 61.707303 | 28,512 |
| 2017-07-14 | 0.393414 | 59.719190 | 28,512 |

The `stage2_optimization_table.csv` contains 15 candidate configurations and
ranks this configuration first among those 15 by mean RMSLE: **0.387690**
(standard deviation 0.006629). The separate
`complete_27_tuning_table.csv` contains 27 candidates but does not include
this Champion, so it does not support a ranking of the Champion among those
27. The closest candidate in the 15-row stage-2 table by mean RMSLE,
`opt_730_l63_samp0806`, has mean RMSLE 0.388015 (standard deviation 0.006463).
The Champion's lead is about 0.000324 RMSLE, while the runner-up's fold
standard deviation is marginally smaller. Thus selection follows the stated
primary rule of lowest mean RMSLE; it is not evidence that the Champion is
best on every fold or most stable by every criterion. The table also shows
that the Champion beats the baseline LightGBM `default` average RMSLE
(0.396710) over the same three folds. The baseline row-level results are in
`reports/lightgbm/Member2_results/experiment_results.csv`.

The evaluation handoff reports these aggregate scores:

| Metric | Value | Scale / aggregation |
|---|---:|---|
| Mean RMSLE | 0.387690 | Arithmetic mean of the three fold RMSLEs; sales scale |
| Mean MAE | 61.424744 | Arithmetic mean of the three fold MAEs; sales scale |
| Pooled RMSLE | 0.387728 | All 85,536 rows pooled; sales scale |
| Pooled MAE | 61.424744 | All 85,536 rows pooled; sales scale |

The individual Champion fold metrics are present in the three run JSON files
listed below. The repository has no persisted aggregate evaluation-result
file for the mean/pooled summary; those four aggregate values are carried over
from the supplied TV4 evaluation handoff rather than recomputed here. The
comparison source is `reports/lightgbm/Member2_results/stage2_optimization_table.csv`.

### Evaluation artifact paths

- Fold 1 predictions and metadata:
  `reports/lightgbm/Member2_results/Checkpoint & Dự báo Validation/validation_predictions/opt_730_l63_mcs35_reg_2017-06-12_val_preds.csv`
  and `reports/lightgbm/Member2_results/Checkpoint & Dự báo Validation/logs/opt_730_l63_mcs35_reg_2017-06-12_run.json`.
- Fold 2 predictions and metadata:
  `reports/lightgbm/Member2_results/Checkpoint & Dự báo Validation/validation_predictions/opt_730_l63_mcs35_reg_2017-06-28_val_preds.csv`
  and `reports/lightgbm/Member2_results/Checkpoint & Dự báo Validation/logs/opt_730_l63_mcs35_reg_2017-06-28_run.json`.
- Fold 3 predictions and metadata:
  `reports/lightgbm/Member2_results/Checkpoint & Dự báo Validation/validation_predictions/opt_730_l63_mcs35_reg_2017-07-14_val_preds.csv`
  and `reports/lightgbm/Member2_results/Checkpoint & Dự báo Validation/logs/opt_730_l63_mcs35_reg_2017-07-14_run.json`.
- Tuning comparison: `reports/lightgbm/Member2_results/stage2_optimization_table.csv`.
- Individual experiment records: `reports/lightgbm/Member2_results/experiment_results.csv`.
- `reports/experiments.csv` is empty (header only) and is not used as results.

The repository's evaluation contract states that target-missing rows are not
filled with zero and prediction keys missing for observed actuals are errors.
The prior actual-label completeness audit in `docs/EVALUATION.md` found a
complete 28,512-key grid for each of the July 15–30 and July 31–August 15
periods in the original `train.csv`; this is an actual-data coverage check,
not evidence of forecast performance on the locked holdout. No locked-holdout
metrics are claimed in this report.

### Error-analysis coverage

The Champion's available saved evaluation artifacts include row-level
predictions and per-fold overall metrics. I found no aggregate error-slice
artifact for this Champion grouped by family, store, horizon, zero/nonzero
actual sales, or historical-sales groups. The existing
`outputs/lgb_fold3_run3/slices.csv` does contain RMSLE slices for family (33
groups), store (54 groups), and horizon (16 groups), but its `run.json` records
a different LightGBM configuration (`feature_set=full`, `train_days=365`,
`n_estimators=300`), not the Champion (`holiday`, 730 days, 1,400 estimators).
Those slice values are therefore not evidence about Champion errors and are
not summarized here. The local error analysis below covers three selected
Champion cases only; aggregate Champion error-slice findings remain
unavailable in the persisted artifacts.

## SHAP global interpretation

The Champion checkpoint for cutoff `2017-07-14` has 27 features. The saved SHAP
sample contains 512 of the 28,512 validation rows. `xai_global.csv` ranks
features by mean absolute SHAP on this sample:

| Rank | Feature | Mean absolute SHAP | Interpretation / feature type |
|---:|---|---:|---|
| 1 | `mean_log_7` | 1.198082935 | Mean log1p sales over the latest 7 historical days; numeric history summary |
| 2 | `mean_log_28` | 0.667732341 | Mean log1p sales over the latest 28 historical days; numeric history summary |
| 3 | `promo_log` | 0.238098762 | log1p of promotion count on the forecast date; numeric known-future covariate |
| 4 | `family` | 0.154469065 | Product-family identifier; categorical identifier, not ordinal |
| 5 | `origin_lag_0` | 0.138422919 | log1p sales on the forecast origin/cutoff; numeric history value |
| 6 | `dow_sin` | 0.112584934 | Sine encoding of weekday cycle; numeric cyclical calendar feature |
| 7 | `store_nbr` | 0.045578804 | Store identifier; categorical identifier, not ordinal |
| 8 | `doy_cos` | 0.032807083 | Cosine encoding of day-of-year cycle; numeric cyclical calendar feature |
| 9 | `origin_lag_6` | 0.023594584 | log1p sales at origin minus 6 days; numeric history value |
| 10 | `origin_lag_13` | 0.023541070 | log1p sales at origin minus 13 days; numeric history value |

Other identity-like/static features (`city`, `state`, `type`, `cluster`) and
`family`/`store_nbr` represent categories or group identity. Their numeric codes
must not be described as ordered quantities. The remaining calendar, holiday,
horizon, promotion, observed-rate and lag/rolling features are numeric or
binary features as defined in `src/store_sales/features.py`.

Mean absolute SHAP measures average attribution magnitude, not direction or a
causal effect. Related predictors (for example, 7/28-day historical means and
origin lags, or promotion variables) can share attribution because they carry
overlapping information. Interpret their individual ranks with that caveat.

Plots and source artifacts:

- [Mean-absolute-SHAP bar chart](../reports/xai/champion_2017-07-14/xai_importance_bar.png)
- [SHAP summary / beeswarm](../reports/xai/champion_2017-07-14/xai_global.png)
- [Global ranking CSV](../reports/xai/champion_2017-07-14/xai_global.csv)
- [Saved sample attributions](../reports/xai/champion_2017-07-14/xai_shap_values.csv)

## Local explanations

The three cases below were selected from the saved 512-row SHAP sample and
joined to validation predictions by `(store_nbr, family, horizon)`; the date
comes from the matched validation row. The sign convention for error is
`prediction - actual`: positive means overprediction, negative means
underprediction. SHAP values and raw predictions below are on the raw
`log1p(sales)` output scale, not sales units. Actuals, predictions and errors
are on the sales scale.

| Case | Store / family | Date / horizon | Actual sales | Predicted sales | Absolute error | Direction | Promotion count |
|---|---|---|---:|---:|---:|---|---:|
| Typical | 44 / PET SUPPLIES | 2017-07-24 / 10 | 26.000 | 30.836 | 4.836 | Over by 4.836 | 0 |
| Promotion | 52 / GROCERY I | 2017-07-20 / 6 | 4,902.000 | 4,886.141 | 15.859 | Under by 15.859 | 79 |
| Large error | 49 / PRODUCE | 2017-07-25 / 11 | 5,380.885 | 6,638.790 | 1,257.905 | Over by 1,257.905 | 11 |

| Case | Base value | SHAP sum | Raw model output |
|---|---:|---:|---:|
| Typical | 3.450673 | 0.009928 | 3.460601 |
| Promotion | 3.450673 | 5.043690 | 8.494363 |
| Large error | 3.450673 | 5.350162 | 8.800836 |

The additivity check is `base value + SHAP sum = raw model output` within
floating-point tolerance. The prediction CSV is on the sales scale after
inverse transformation; do not compare it directly to the raw output value.

### Technical audit evidence

The saved explanation contains 512 sampled validation rows and 27 features;
it does not explain all 28,512 validation rows. Feature names and their order
match the Champion checkpoint schema exactly. The saved SHAP values contain no
NaN or infinite values. Against raw LightGBM predictions on the same feature
matrix, the maximum absolute additivity error is `5.33e-14` and the mean
absolute additivity error is `1.26e-14`, both below the audit tolerance of
`1e-6`. SHAP explains the raw LightGBM output on the `log1p(sales)` scale,
before inverse transformation to sales. The original Champion checkpoint was
not modified; the XAI copy's LF-normalized SHA256 matches the original
checkpoint (`B23903AA588187C09B04388A05C67252C24E53B105A16EFFCB6BD05C38AF7577`).

### Typical case

The absolute error (4.836 sales) is near the median absolute error (5.311)
among the sampled validation rows, and this row has no promotion. Its largest
positive contributions are `store_nbr` (+0.100511), `mean_log_7` (+0.095115),
and `origin_lag_0` (+0.066672); largest negative contributions are `dow_sin`
(−0.098612), `promo_log` (−0.067580), and `doy_cos` (−0.035948). The feature
attributions shift the model output slightly above the base value, yielding a
sales forecast modestly above actual. The store identifier contribution is a
categorical model association, not a numeric store-order effect.

[Typical-case waterfall](../reports/xai/champion_2017-07-14/selected_cases/typical_store44_pet_supplies_h10_2017-07-24.png)

### Promotion case

`promo_log` contributes +0.788587 raw log units for a row with 79 promotions;
`mean_log_7` (+1.911542), `mean_log_28` (+1.784470), and `origin_lag_0`
(+0.393158) also push the output upward. The largest negative contribution is
`dow_sin` (−0.107810). The forecast is 15.859 sales below actual, despite the
positive promotion attribution. This illustrates that a positive feature
contribution does not guarantee an overprediction or determine the complete
forecast by itself.

[Promotion-case waterfall](../reports/xai/champion_2017-07-14/selected_cases/promotion_store52_grocery_i_h06_2017-07-20.png)

### Large-error case

The model overpredicts by 1,257.905 sales. The strongest positive attributions
are `mean_log_28` (+2.012196), `mean_log_7` (+1.993036), `promo_log`
(+0.559921), `origin_lag_0` (+0.412951), and `family` (+0.332471). The strongest
negative contributions are `dow_sin` (−0.122139), `doy_cos` (−0.042115), and
`observed_28` (−0.007798). Historical sales summaries dominate this local
explanation; these attributions describe the model output and do not identify
why actual sales differed.

[Large-error waterfall](../reports/xai/champion_2017-07-14/selected_cases/large_error_store49_produce_h11_2017-07-25.png)

## How Evaluation and XAI complement each other

Evaluation quantifies predictive performance across the three temporal folds,
using row-aligned actual/prediction pairs and sales-scale metrics. SHAP
complements those aggregate metrics by showing which inputs the selected
Champion relied on globally in a fixed validation sample and how the model
assembled raw output for three keyed cases. The large-error case links an
individual prediction miss to its model attributions; it does not replace
error analysis across all rows. Global and local SHAP therefore add
interpretability to, but do not validate or supersede, the fold metrics.

## Limitations and unresolved evidence

- Global SHAP importance is based on 512 sampled rows, not all 28,512 fold-3
  validation rows. Local cases also come from that sample.
- `store_nbr`, `family`, and other category identifiers are not ordered
  numerical variables. Waterfall labels show numeric codes for some identity
  features; interpret those values as category encodings only.
- Correlated or related features can share attribution; individual feature
  ranks and case attributions are not independent effects.
- SHAP explains model associations and does not establish causality.
- The XAI `run.json` records `git_dirty=true`; the checkpoint/artifacts were
  produced from a working tree with uncommitted changes at run time.
- `run.json` fingerprints `train.csv`, `test.csv`, and `stores.csv`, but does
  not fingerprint `holidays_events.csv`. Exact holiday-input provenance is
  therefore not fully reproducible from that manifest alone.
- No aggregate evaluation JSON/CSV was found in the inspected report paths.
  The fold metrics are persisted in the per-fold run logs; mean/pooled metrics
  above are the supplied TV4 handoff values. The repository's
  `reports/experiments.csv` is empty.
- No locked internal holdout metrics are included. The documented holdout
  window is separate from the three tuning folds.

## Artifact inventory and reproducibility references

The Champion model used for SHAP is stored at
`reports/xai/champion_2017-07-14/lightgbm.txt`, with metadata in
`reports/xai/champion_2017-07-14/run.json` and matched validation predictions
in `reports/xai/champion_2017-07-14/validation_predictions.csv`. The report
does not alter the model, predictions or SHAP values.

Evaluation protocol, keyed prediction requirements and actual-label coverage
audit: [EVALUATION.md](EVALUATION.md). Temporal fold and holdout definitions:
[THUC_NGHIEM.md](THUC_NGHIEM.md).
