"""At origin t all historical features use dates <= t; known future promos use t+h."""

import numpy as np
import pandas as pd


def window(panel, origin, lookback, horizon):
    t = panel.position(origin)
    if lookback < 28 or t < lookback - 1 or t + horizon >= len(panel.dates):
        raise ValueError("Insufficient history/future calendar; lookback must be >=28")
    raw = panel.sales[t - lookback + 1 : t + 1].T
    observed = np.isfinite(raw)
    # Missing calendar rows are a modelling placeholder, paired with an explicit flag.
    hist = np.stack(
        [
            np.log1p(np.nan_to_num(raw)),
            np.log1p(np.nan_to_num(panel.promo[t - lookback + 1 : t + 1].T)),
            observed.astype("float32"),
        ],
        axis=-1,
    ).astype("float32")
    future_dates = panel.dates[t + 1 : t + horizon + 1]
    future_promo = panel.promo[t + 1 : t + horizon + 1].T
    if not np.isfinite(future_promo).all():
        raise ValueError("Missing future promotions: choose and document a policy first")
    dow = np.asarray(future_dates.dayofweek)
    doy = np.asarray(future_dates.dayofyear)
    calendar = np.stack(
        [
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
            np.arange(1, horizon + 1) / horizon,
        ],
        axis=-1,
    )
    future = np.concatenate(
        [
            np.broadcast_to(calendar, (len(panel.groups), horizon, 5)),
            np.log1p(future_promo)[..., None],
        ],
        axis=-1,
    ).astype("float32")
    target = panel.sales[t + 1 : t + horizon + 1].T.copy()
    return hist, future, target


def origin_dates(panel, cutoff, config):
    end = panel.position(cutoff) - config["horizon"]
    start = max(config["lookback"] - 1, end - config["train_days"] + 1)
    positions = list(range(end, start - 1, -config["stride"]))[::-1]
    # Historical missing calendar rows can also remove future promotion records.
    # Skip such supervised origins rather than invent future exogenous inputs.
    positions = [
        t for t in positions if np.isfinite(panel.promo[t + 1 : t + config["horizon"] + 1]).all()
    ]
    if not positions:
        raise ValueError("No supervised training windows before cutoff")
    return [panel.dates[x] for x in positions]


"""

def tabular(panel, hist, future, feature_set="full"):
    groups, horizon, _ = future.shape
    frame = pd.DataFrame(np.repeat(panel.static, horizon, axis=0), columns=panel.static_names)
    for i, name in enumerate(["dow_sin", "dow_cos", "doy_sin", "doy_cos", "horizon", "promo_log"]):
        frame[name] = future[:, :, i].reshape(-1)
    if feature_set != "no_history":
        # These are relative to the forecast ORIGIN, not to each future target date.
        for lag in [0, 6, 13, 27]:
            frame[f"origin_lag_{lag}"] = np.repeat(hist[:, -lag-1, 0], horizon)
        for days in [7, 28]:
            frame[f"mean_log_{days}"] = np.repeat(hist[:, -days:, 0].mean(axis=1), horizon)
            frame[f"observed_{days}"] = np.repeat(hist[:, -days:, 2].mean(axis=1), horizon)
    if feature_set == "no_promo":
        frame = frame.drop(columns="promo_log")
    return frame.astype("float32")
"""
VALID_FEATURE_SETS = {"full", "no_history", "no_promo", "holiday", "oil", "holiday_oil"}


def tabular(panel, hist, future, feature_set="full", aux=None):
    if feature_set not in VALID_FEATURE_SETS:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    groups, horizon, _ = future.shape
    frame = pd.DataFrame(np.repeat(panel.static, horizon, axis=0), columns=panel.static_names)
    for i, name in enumerate(["dow_sin", "dow_cos", "doy_sin", "doy_cos", "horizon", "promo_log"]):
        frame[name] = future[:, :, i].reshape(-1)
    if feature_set != "no_history":
        # These are relative to the forecast ORIGIN, not to each future target date.
        for lag in [0, 6, 13, 27]:
            frame[f"origin_lag_{lag}"] = np.repeat(hist[:, -lag - 1, 0], horizon)
        for days in [7, 28]:
            frame[f"mean_log_{days}"] = np.repeat(hist[:, -days:, 0].mean(axis=1), horizon)
            frame[f"observed_{days}"] = np.repeat(hist[:, -days:, 2].mean(axis=1), horizon)
    if feature_set == "no_promo":
        frame = frame.drop(columns="promo_log")
    if aux is not None:
        for col, values in aux.items():
            frame[col] = values
    return frame.astype("float32")


def seasonal_naive(panel, origin, horizon):
    t = panel.position(origin)
    history = panel.sales[max(0, t - 55) : t + 1]
    history_dates = panel.dates[max(0, t - 55) : t + 1]
    predictions = []
    for date in panel.dates[t + 1 : t + horizon + 1]:
        match = history[np.asarray(history_dates.dayofweek) == date.dayofweek][-4:]
        valid = np.isfinite(match)
        mean = np.divide(
            np.nansum(match, axis=0),
            valid.sum(axis=0),
            out=np.zeros(history.shape[1]),
            where=valid.sum(axis=0) > 0,
        )
        predictions.append(mean)
    return np.asarray(predictions, dtype="float32").T
