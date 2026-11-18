# Evaluation Framework

The shared interface is in `store_sales.evaluate`. It accepts keyed validation
predictions from the baseline, LightGBM, or LSTM; it does not train or generate
predictions for any model.

## Validation prediction CSV contract

Use one row per `(date, store_nbr, family)` and these columns:

| Column | Required | Meaning |
|---|---|---|
| `date` | Yes | Forecast target date, parseable as a date |
| `store_nbr` | Yes | Store key, same type/value convention as `train.csv` |
| `family` | Yes | Product-family key, same convention as `train.csv` |
| `prediction` | Yes | Predicted `sales`, finite and nonnegative; keep decimals |
| `id` | Recommended | ID copied from the corresponding actual/competition row |

For a scored validation fold, include rows with an observed `sales` label and
write predictions in the same row order. Missing actual targets are excluded
from scoring; do not add rows with sales set to zero. The built-in panel
`score()` path generates a full-panel forecast, then aligns it to observed
validation labels by key. Sort by `date`, then `store_nbr`, then
`family`, matching the actual frame used for scoring. Dates must match the
intended fold range. Do not round, clip silently, impute missing targets, or
include training rows. The `id` column is optional for keyed validation because
the composite keys identify each row; when supplied, it must be unique and
match actual IDs in the same order. Competition `submission.csv` remains the
separate strict `id,sales` format validated by `validate_submission`.

Example (illustrative format only; not an experiment result):

```csv
date,store_nbr,family,id,prediction
2017-06-13,1,BEVERAGES,1001,12.75
2017-06-13,1,BREAD/BAKERY,1002,3.20
```

## Python interface

```python
from store_sales.evaluate import evaluate_predictions, write_evaluation

result = evaluate_predictions(
    actual_frame, prediction_frame, model="lightgbm", fold="tuning_1",
    origin="2017-06-12",
)
write_evaluation(result, "outputs/lgb_tuning_1")
```

`actual_frame` needs `date,store_nbr,family,sales`; prediction frame needs the
same keys plus `prediction`. `origin` checks that the dates are exactly
origin+1 through the fold horizon. The evaluator rejects duplicate keys/IDs,
missing columns/values, NaN/Inf, negative actuals or predictions, missing/extra
rows, key or ID order differences, and incorrect/gapped date ranges.

Each evaluation returns overall RMSLE, MAE, and row count; RMSLE and MAE by
family, store, and horizon; and row-level errors. `compare_models` creates a
summary from these evaluated results. Only compare results produced from the
same actual fold and row keys. `error_analysis(result)` returns the largest
absolute-error cases for review. These utilities do not select a winning model
or establish causal explanations.

The panel `score()` output also reports `n_forecast_expected` (series × horizon),
`n_actual_observed`, `n_target_missing`, `n` (rows actually scored), and
`target_coverage_pct = 100 × n / n_forecast_expected`. Prediction keys missing
for an observed actual are an error; absent actual targets reduce coverage and
are never replaced with zero.

## Actual-label completeness audit

Audited `data/raw/train.csv` by `(date, store_nbr, family)` on 2026-11-18.
The file has 3,000,888 rows and the two 16-day windows contain a complete
1,782-series grid:

| Window | Actual keys | Expected keys | Missing keys | Coverage |
|---|---:|---:|---:|---:|
| 2017-07-15..2017-07-30 | 28,512 | 28,512 | 0 | 100% |
| 2017-07-31..2017-08-15 | 28,512 | 28,512 | 0 | 100% |
| Combined | 57,024 | 57,024 | 0 | 100% |

Each date has 1,782 rows. There are no duplicate forecast keys, duplicate IDs,
blank sales, or invalid/negative sales in these windows. Zero sales are present
(4,202 in the first window and 4,156 in the second) and remain valid actuals.
The supplied EDA row count, 2,943,864, is exactly 57,024 below the raw train
count; the removed rows correspond to these two windows. Use the original
`train.csv` for scoring, not the EDA subset. This is a data completeness audit,
not a model result: predictions are still needed before a forecast metric can
be reported.

The repository experiment plan currently labels 2017-07-15..2017-07-30 as
tuning fold 3 and 2017-07-31..2017-08-15 as the locked internal holdout. Keep
those roles distinct in experiment records even if the first interval was
called an illustrative holdout elsewhere.

## Awaiting team handoffs

- **TV1:** confirmed real-data date/key quality review and any agreed changes to
  the data contract.
- **TV2:** LightGBM and baseline fold predictions, with model/config/fold
  provenance; baseline's existing CLI score remains supported.
- **TV3:** LSTM fold predictions and model/config/fold provenance.
- Model comparison tables, real fold results, final holdout scoring, and XAI
  case studies require those handoffs. No experimental predictions or scores
  are included here.
