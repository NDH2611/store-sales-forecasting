"""Auxiliary tables. They are not automatically fed into forecasting models."""
import numpy as np
import pandas as pd


HOLIDAY_FLAGS = [
    "is_holiday", "is_transfer", "is_additional", "is_bridge",
    "is_work_day", "is_event", "is_transferred_original",
]


def build_holiday_features(holidays, stores, start, end):
    events = holidays.copy()
    locations = stores[["store_nbr", "city", "state"]].copy()

    events["date"] = pd.to_datetime(events["date"], errors="raise")
    required = ["date", "type", "locale", "locale_name", "transferred"]

    if events[required].isna().any().any() or locations.isna().any().any():
        raise ValueError("Missing holiday/store metadata")

    if not locations["store_nbr"].is_unique:
        raise ValueError("Duplicate store_nbr")

    if not events["transferred"].isin([True, False]).all():
        raise ValueError("transferred must contain booleans")

    if not events["locale"].isin(["National", "Regional", "Local"]).all():
        raise ValueError("Unknown holiday locale")

    event_types = [
        "Holiday", "Transfer", "Additional",
        "Bridge", "Work Day", "Event",
    ]

    if not events["type"].isin(event_types).all():
        raise ValueError("Unknown holiday type")

    for scope, column in [("Local", "city"), ("Regional", "state")]:
        names = set(
            events.loc[events["locale"].eq(scope), "locale_name"]
        )
        if names - set(locations[column]):
            raise ValueError(f"Unmatched {scope} holiday names")

    dates = pd.date_range(start, end, freq="D")

    if dates.empty:
        raise ValueError("Empty date range")

    events = events.loc[
        events["date"].between(dates[0], dates[-1])
    ]

    parts = [
        events.loc[events["locale"].eq("National")].merge(
            locations,
            how="cross",
        )
    ]

    for scope, column in [("Local", "city"), ("Regional", "state")]:
        parts.append(
            events.loc[events["locale"].eq(scope)].merge(
                locations,
                left_on="locale_name",
                right_on=column,
                how="inner",
            )
        )

    expanded = pd.concat(parts, ignore_index=True)
    active = ~expanded["transferred"].astype(bool)

    expanded["is_holiday"] = (
        active
        & expanded["type"].isin(
            ["Holiday", "Transfer", "Additional", "Bridge"]
        )
    ).astype("uint8")

    for flag, kind in zip(HOLIDAY_FLAGS[1:6], event_types[1:]):
        expanded[flag] = (
            active & expanded["type"].eq(kind)
        ).astype("uint8")

    expanded["is_transferred_original"] = (
        expanded["type"].eq("Holiday") & ~active
    ).astype("uint8")

    flags = (
        expanded.groupby(
            ["date", "store_nbr"],
            as_index=False,
        )[HOLIDAY_FLAGS]
        .max()
    )

    calendar = pd.DataFrame({"date": dates}).merge(
        locations[["store_nbr"]],
        how="cross",
    )

    result = calendar.merge(
        flags,
        on=["date", "store_nbr"],
        how="left",
        validate="one_to_one",
    )

    result[HOLIDAY_FLAGS] = (
        result[HOLIDAY_FLAGS].fillna(0).astype("uint8")
    )

    return result


def build_oil_features(oil, start, end, cutoff):
    prices = oil[["date", "dcoilwtico"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="raise")

    if prices["date"].isna().any() or not prices["date"].is_unique:
        raise ValueError("Missing or duplicate oil date")

    cutoff = pd.Timestamp(cutoff)

    # Bỏ giá tương lai trước khi kiểm tra và điền giá.
    known = (
        prices.loc[prices["date"] <= cutoff]
        .set_index("date")["dcoilwtico"]
    )

    known = pd.to_numeric(known, errors="raise").sort_index()
    observed = known.dropna()

    if not np.isfinite(observed).all() or observed.lt(0).any():
        raise ValueError("Invalid historical oil price")

    dates = pd.date_range(start, end, freq="D")

    if dates.empty or pd.isna(cutoff):
        raise ValueError("Invalid date range/cutoff")

    # Giữ được giá quá khứ ngay cả khi start nằm sau ngày có giá.
    combined_dates = known.index.union(dates).sort_values()

    filled = (
        known.reindex(combined_dates)
        .ffill()
        .reindex(dates)
    )

    result = pd.DataFrame({
        "date": dates,
        "oil_price": filled.to_numpy(),
    })

    result["oil_unobserved_at_origin"] = (
        known.reindex(dates)
        .isna()
        .to_numpy()
        .astype("uint8")
    )

    result["oil_still_missing"] = (
        result["oil_price"].isna().astype("uint8")
    )

    result["oil_is_forecast"] = (
        dates > cutoff
    ).astype("uint8")

    return result