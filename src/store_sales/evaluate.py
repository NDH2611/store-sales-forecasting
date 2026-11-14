import numpy as np
import pandas as pd


def rmsle(y_true, y_pred):
    true, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape or not true.size:
        raise ValueError("RMSLE requires nonempty matching shapes")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Non-finite values")
    if (true < 0).any() or (pred < 0).any():
        raise ValueError("RMSLE requires nonnegative values")
    return float(np.sqrt(np.mean((np.log1p(pred)-np.log1p(true))**2)))


def prediction_frame(panel, origin, predictions):
    horizon = predictions.shape[1]
    dates = panel.dates[panel.position(origin)+1:panel.position(origin)+1+horizon]
    keys = panel.groups.to_frame(index=False).loc[
        panel.groups.to_frame(index=False).index.repeat(horizon)
    ].reset_index(drop=True)
    keys["date"] = np.tile(dates, len(panel.groups))
    keys["prediction"] = np.asarray(predictions).reshape(-1)
    return keys


def score(panel, origin, predictions, output):
    expected = panel.train.loc[(panel.train.date > pd.Timestamp(origin)) &
                              (panel.train.date <= pd.Timestamp(origin)+pd.Timedelta(days=predictions.shape[1]))]
    if expected.empty:
        raise ValueError("No validation labels at this origin")
    frame = expected.merge(prediction_frame(panel, origin, predictions),
                           on=["date", "store_nbr", "family"], validate="one_to_one")
    if len(frame) != len(expected):
        raise ValueError("Validation keys do not match")
    frame["horizon"] = (frame.date-pd.Timestamp(origin)).dt.days
    frame.to_csv(output / "validation_predictions.csv", index=False)
    slices = []
    for col in ["family", "store_nbr", "horizon"]:
        for value, part in frame.groupby(col):
            slices.append({"slice": col, "value": value, "n": len(part),
                           "rmsle": rmsle(part.sales, part.prediction)})
    pd.DataFrame(slices).to_csv(output / "slices.csv", index=False)
    return {"rmsle": rmsle(frame.sales, frame.prediction),
            "mae": float(np.abs(frame.sales-frame.prediction).mean()), "n": len(frame)}


def validate_submission(submission, test):
    if list(submission.columns) != ["id", "sales"]:
        raise ValueError("Submission columns must be id,sales in that order")
    if len(submission) != len(test) or submission.id.duplicated().any():
        raise ValueError("Wrong row count or duplicate id")
    if not submission.id.reset_index(drop=True).equals(test.id.reset_index(drop=True)):
        raise ValueError("Submission ids/order differ from test")
    if not np.isfinite(submission.sales.to_numpy()).all() or (submission.sales < 0).any():
        raise ValueError("Submission sales must be finite and nonnegative")


def submission(panel, origin, predictions, output):
    frame = panel.test.merge(prediction_frame(panel, origin, predictions),
                             on=["date", "store_nbr", "family"], how="left", validate="one_to_one")
    result = frame[["id", "prediction"]].rename(columns={"prediction": "sales"})
    validate_submission(result, panel.test)
    result.to_csv(output / "submission.csv", index=False)
    return result
