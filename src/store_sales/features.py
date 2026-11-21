"""At origin t all historical features use dates <= t; known future promos use t+h."""

import numpy as np
import pandas as pd


def window(panel, origin, lookback, horizon):
    t = panel.position(origin)
    if lookback < 56 or t < lookback - 1 or t + horizon >= len(panel.dates):
        raise ValueError("Insufficient history/future calendar; lookback must be >=56")
    raw = panel.sales[t - lookback + 1 : t + 1].T
    observed = np.isfinite(raw)
    hist = np.stack(
        [
            np.log1p(np.nan_to_num(raw)),
            np.log1p(np.nan_to_num(panel.promo[t - lookback + 1 : t + 1].T)),
            observed.astype("float32"),
        ],
        axis=-1,
    ).astype("float32")

    future_dates = panel.dates[t + 1 : t + horizon + 1]
    # Lấy promo từ t-7 đến t+horizon để tính được lag/lead/rolling quanh ngày dự báo
    ext_promo = np.log1p(np.nan_to_num(panel.promo[t - 6 : t + horizon + 1].T))
    future_promo = panel.promo[t + 1 : t + horizon + 1].T
    if not np.isfinite(future_promo).all():
        raise ValueError("Missing future promotions: choose and document a policy first")

    dow = np.asarray(future_dates.dayofweek)
    doy = np.asarray(future_dates.dayofyear)
    day = np.asarray(future_dates.day)
    is_payday = ((day == 15) | np.asarray(future_dates.is_month_end)).astype("float32")
    days_since_payday = np.where(day >= 15, day - 15, day).astype("float32")

    calendar = np.stack(
        [
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
            np.arange(1, horizon + 1) / horizon,
            day / 31.0,
            is_payday,
            days_since_payday / 15.0,
        ],
        axis=-1,
    )

    n_groups = len(panel.groups)
    promo_now = np.log1p(future_promo)
    # ext_promo index 7 tương ứng với t+1
    promo_lag1 = ext_promo[:, 6 : 6 + horizon]
    promo_roll7 = np.stack(
        [ext_promo[:, h : h + 7].mean(axis=1) for h in range(1, horizon + 1)], axis=1
    )

    future = np.concatenate(
        [
            np.broadcast_to(calendar, (n_groups, horizon, 8)),
            promo_now[..., None],
            promo_lag1[..., None],
            promo_roll7[..., None],
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


VALID_FEATURE_SETS = {"full", "no_history", "no_promo", "holiday", "oil", "holiday_oil"}


def tabular(panel, hist, future, feature_set="full", aux=None):
    if feature_set not in VALID_FEATURE_SETS:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    groups, horizon, _ = future.shape
    frame = pd.DataFrame(np.repeat(panel.static, horizon, axis=0), columns=panel.static_names)

    # 1. Ánh xạ chính xác 11 kênh từ mảng future được tạo bởi hàm window()
    future_cols = [
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "horizon",
        "day_norm",
        "is_payday",
        "days_since_payday",
        "promo_log",
        "promo_lag1",
        "promo_roll7",
    ]
    for i, name in enumerate(future_cols):
        frame[name] = future[:, :, i].reshape(-1)

    # 2. Tính bổ sung promo_lead1 và promo_roll3 từ đúng kênh promo_log (index 8) và promo_lag1 (index 9)
    promo_2d = future[:, :, 8]
    promo_lag1 = future[:, :, 9]
    promo_lead1 = np.concatenate([promo_2d[:, 1:], promo_2d[:, -1:]], axis=1)
    promo_roll3 = (promo_lag1 + promo_2d + promo_lead1) / 3.0
    frame["promo_lead1"] = promo_lead1.reshape(-1)
    frame["promo_roll3"] = promo_roll3.reshape(-1)

    # 3. Các đặc trưng lịch sử nhân quả (chỉ dùng dữ liệu <= origin t)
    if feature_set != "no_history":
        # Origin lags tại cutoff t
        for lag in [0, 1, 6, 13, 27]:
            frame[f"origin_lag_{lag}"] = np.repeat(hist[:, -lag - 1, 0], horizon)

        # Same-DOW lags (đúng thứ trong tuần của từng ngày mục tiêu t+h)
        lookback = hist.shape[1]
        for week in [1, 2, 3, 4]:
            same_dow = np.zeros((groups, horizon), dtype=np.float32)
            for h in range(horizon):
                step_back = 7 * week - (h + 1)
                while step_back < 0:
                    step_back += 7
                same_dow[:, h] = hist[:, lookback - 1 - step_back, 0]
            frame[f"same_dow_w{week}"] = same_dow.reshape(-1)

        # Rolling statistics (Mean, Std, Zero-rate, Observed-rate)
        for days in [7, 14, 28, 56]:
            d = min(days, lookback)
            sl = hist[:, -d:, 0]
            frame[f"mean_log_{days}"] = np.repeat(sl.mean(axis=1), horizon)
            frame[f"std_log_{days}"] = np.repeat(sl.std(axis=1), horizon)
            frame[f"zero_rate_{days}"] = np.repeat((sl == 0).mean(axis=1), horizon)
            frame[f"observed_{days}"] = np.repeat(hist[:, -d:, 2].mean(axis=1), horizon)

    # 4. Loại bỏ toàn bộ đặc trưng khuyến mãi khi chạy ablation no_promo
    if feature_set == "no_promo":
        frame = frame.drop(
            columns=["promo_log", "promo_lag1", "promo_lead1", "promo_roll3", "promo_roll7"]
        )

    # 5. Gắn thêm các biến ngoại sinh (ngày lễ, giá dầu...) nếu có
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
