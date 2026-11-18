"""Shared validation, scoring, model comparison, and error analysis utilities."""

from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["date", "store_nbr", "family"]


def rmsle(y_true, y_pred):
    """Root mean squared logarithmic error for nonnegative, finite values."""
    true, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape or not true.size:
        raise ValueError("RMSLE requires nonempty matching shapes")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("RMSLE requires finite values")
    if (true < 0).any() or (pred < 0).any():
        raise ValueError("RMSLE requires nonnegative values")
    return float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(true)) ** 2)))


def validate_predictions(actual, predictions, *, origin=None, expected_ids=False):
    """Validate keyed prediction rows against actuals, preserving strict row order.

    Actuals use ``sales`` and predictions use ``prediction``. Both must contain
    date/store_nbr/family. If actuals contain IDs, prediction IDs are checked too.
    """
    actual_required = KEYS + ["sales"]
    prediction_required = KEYS + ["prediction"]
    actual, predictions = actual.copy(), predictions.copy()
    for name, frame, required in (("actual", actual, actual_required),
                                  ("predictions", predictions, prediction_required)):
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        frame["date"] = pd.to_datetime(frame.date, errors="raise")
        if frame[required].isna().any().any():
            raise ValueError(f"{name} contains NaN/null values")
        if "id" in frame and frame.id.isna().any():
            raise ValueError(f"{name} contains missing IDs")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{name} contains duplicate forecast keys")
        if "id" in frame and frame.id.duplicated().any():
            raise ValueError(f"{name} contains duplicate IDs")
        values = frame["sales" if name == "actual" else "prediction"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
        if (values < 0).any():
            raise ValueError(f"{name} contains negative values")

    if len(actual) != len(predictions):
        raise ValueError("Prediction row count differs from actuals (missing or extra IDs/keys)")
    if not actual[KEYS].reset_index(drop=True).equals(
        predictions[KEYS].reset_index(drop=True)
    ):
        raise ValueError("Prediction keys/order differ from actuals")
    if "id" in actual:
        if "id" not in predictions:
            if expected_ids:
                raise ValueError("Predictions are missing IDs")
        elif not actual.id.reset_index(drop=True).equals(predictions.id.reset_index(drop=True)):
            raise ValueError("Prediction IDs/order differ from actuals")
    elif expected_ids:
        raise ValueError("Actuals do not contain IDs to validate")

    actual_dates = pd.DatetimeIndex(actual.date.unique())
    pred_dates = pd.DatetimeIndex(predictions.date.unique())
    if not actual_dates.is_monotonic_increasing or not pred_dates.is_monotonic_increasing:
        raise ValueError("Forecast dates must be in ascending order")
    if not actual_dates.equals(pred_dates):
        raise ValueError("Prediction date range differs from actuals")
    if len(actual_dates) > 1 and not np.all(
        np.diff(actual_dates.asi8) == pd.Timedelta(days=1).value
    ):
        raise ValueError("Validation dates must be a continuous daily range")
    if origin is not None:
        origin = pd.Timestamp(origin).normalize()
        expected = pd.date_range(origin + pd.Timedelta(days=1), periods=len(actual_dates), freq="D")
        if not actual_dates.equals(expected):
            raise ValueError("Validation dates do not match the requested origin/horizon")
    return True


def evaluate_predictions(actual, predictions, *, model="model", fold=None, origin=None):
    """Score one model on keyed rows and return metrics, slices, and row errors."""
    validate_predictions(actual, predictions, origin=origin)
    frame = actual.copy().reset_index(drop=True)
    frame["prediction"] = predictions.prediction.to_numpy(dtype=float)
    if origin is None:
        first_date = pd.to_datetime(frame.date).min()
        frame["horizon"] = (pd.to_datetime(frame.date) - first_date).dt.days + 1
    else:
        frame["horizon"] = (pd.to_datetime(frame.date) - pd.Timestamp(origin)).dt.days
    frame["error"] = frame.prediction - frame.sales
    frame["absolute_error"] = frame.error.abs()
    frame["squared_log_error"] = (np.log1p(frame.prediction) - np.log1p(frame.sales)) ** 2
    overall = {"model": model, "fold": fold, "rmsle": rmsle(frame.sales, frame.prediction),
               "mae": float(frame.absolute_error.mean()), "n": len(frame)}
    slices = []
    for column in ("family", "store_nbr", "horizon"):
        for value, part in frame.groupby(column, sort=True):
            slices.append({"model": model, "fold": fold, "slice": column, "value": value,
                           "n": len(part), "rmsle": rmsle(part.sales, part.prediction),
                           "mae": float(part.absolute_error.mean())})
    return {"metrics": overall, "slices": pd.DataFrame(slices), "rows": frame}


def compare_models(results):
    """Build a comparable summary from ``{model_name: evaluation_result}`` results."""
    if not results:
        raise ValueError("At least one evaluated model is required")
    return pd.DataFrame([result["metrics"] for result in results.values()]).sort_values(
        ["rmsle", "model"], ignore_index=True
    )


def error_analysis(result, worst_n=20):
    """Return largest absolute-error cases first, retaining keys and diagnostics."""
    if worst_n < 1:
        raise ValueError("worst_n must be positive")
    return result["rows"].sort_values(
        ["absolute_error", "date", "store_nbr", "family"],
        ascending=[False, True, True, True], ignore_index=True,
    ).head(worst_n)


def write_evaluation(result, output):
    """Persist reviewable validation rows, metric slices, and error cases."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result["rows"].to_csv(output / "validation_predictions.csv", index=False)
    result["slices"].to_csv(output / "slices.csv", index=False)
    error_analysis(result).to_csv(output / "error_analysis.csv", index=False)


def prediction_frame(panel, origin, predictions):
    horizon = predictions.shape[1]
    dates = panel.dates[panel.position(origin) + 1:panel.position(origin) + 1 + horizon]
    group_frame = panel.groups.to_frame(index=False)
    keys = group_frame.loc[np.tile(group_frame.index, horizon)].reset_index(drop=True)
    keys["date"] = np.repeat(dates, len(panel.groups))
    keys["prediction"] = np.asarray(predictions).T.reshape(-1)
    return keys


def score(panel, origin, predictions, output):
    expected = panel.train.loc[(panel.train.date > pd.Timestamp(origin)) &
                               (panel.train.date <= pd.Timestamp(origin) +
                                pd.Timedelta(days=predictions.shape[1]))].copy()
    if expected.empty:
        raise ValueError("No validation labels at this origin")
    predicted = prediction_frame(panel, origin, predictions)
    values = predicted.prediction.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Validation predictions must be finite and nonnegative")
    if predicted.duplicated(KEYS).any():
        raise ValueError("Validation predictions contain duplicate keys")
    # Forecasts cover the full panel, while the raw validation labels may omit
    # unobserved targets. Score only observed labels without imputing missing sales.
    aligned = expected.merge(predicted[KEYS + ["prediction"]], on=KEYS, how="left",
                             validate="one_to_one", sort=False, indicator=True)
    if (aligned._merge != "both").any():
        raise ValueError("Validation predictions are missing keys with observed actuals")
    aligned = aligned.drop(columns="_merge")
    if not aligned[KEYS].reset_index(drop=True).equals(
        expected[KEYS].reset_index(drop=True)
    ):
        raise ValueError("Key alignment changed validation row order")
    result = evaluate_predictions(aligned, aligned[KEYS + ["prediction"]],
                                  model="baseline", origin=origin)
    n_expected = len(predicted)
    n_scored = result["metrics"]["n"]
    result["metrics"].update({
        "n_forecast_expected": n_expected,
        "n_actual_observed": len(expected),
        "n_target_missing": n_expected - len(expected),
        "target_coverage": n_scored / n_expected,
        "target_coverage_pct": 100.0 * n_scored / n_expected,
    })
    write_evaluation(result, output)
    return result["metrics"].copy()


def validate_submission(submission, test):
    if list(submission.columns) != ["id", "sales"]:
        raise ValueError("Submission columns must be id,sales in that order")
    if submission.id.isna().any() or test.id.isna().any():
        raise ValueError("Submission/test IDs must not be missing")
    if submission.id.duplicated().any() or test.id.duplicated().any():
        raise ValueError("Submission/test contains duplicate IDs")
    if len(submission) != len(test):
        raise ValueError("Submission row count differs from test")
    if not submission.id.reset_index(drop=True).equals(test.id.reset_index(drop=True)):
        raise ValueError("Submission IDs/order differ from test")
    if not np.isfinite(submission.sales.to_numpy(dtype=float)).all() or (submission.sales < 0).any():
        raise ValueError("Submission sales must be finite and nonnegative")


def submission(panel, origin, predictions, output):
    frame = panel.test.merge(prediction_frame(panel, origin, predictions),
                             on=KEYS, how="left", validate="one_to_one")
    result = frame[["id", "prediction"]].rename(columns={"prediction": "sales"})
    validate_submission(result, panel.test)
    result.to_csv(Path(output) / "submission.csv", index=False)
    return result
