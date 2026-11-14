"""Load a daily panel. Missing records are flagged, never silently treated as real sales."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["date", "store_nbr", "family"]


def validate_rows(frame, target=False):
    required = KEY + ["id", "onpromotion"] + (["sales"] if target else [])
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame[required].isna().any().any():
        raise ValueError("Null values in required raw columns; inspect before imputing")
    if frame.duplicated(KEY).any() or frame.id.duplicated().any():
        raise ValueError("Duplicate series/date or id")
    for col in ["onpromotion"] + (["sales"] if target else []):
        if not np.isfinite(frame[col].to_numpy(dtype=float)).all() or (frame[col] < 0).any():
            raise ValueError(f"Invalid {col}")


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    groups: pd.MultiIndex
    sales: np.ndarray
    promo: np.ndarray
    static: np.ndarray
    static_names: list
    store_codes: np.ndarray
    family_codes: np.ndarray
    train: pd.DataFrame
    test: pd.DataFrame
    quality: dict

    def position(self, date):
        return self.dates.get_loc(pd.Timestamp(date))


def load_panel(directory):
    directory = Path(directory)
    train = pd.read_csv(directory / "train.csv", parse_dates=["date"])
    test = pd.read_csv(directory / "test.csv", parse_dates=["date"])
    validate_rows(train, target=True)
    validate_rows(test)
    if train.date.max() >= test.date.min():
        raise ValueError("Train/test dates overlap")
    groups = pd.MultiIndex.from_frame(train[["store_nbr", "family"]].drop_duplicates()).sort_values()
    test_groups = pd.MultiIndex.from_frame(test[["store_nbr", "family"]].drop_duplicates())
    if len(test_groups.difference(groups)):
        raise ValueError("Unseen test series: implement a cold-start policy first")
    dates = pd.date_range(train.date.min(), test.date.max(), freq="D")

    def matrix(frame, col):
        return frame.pivot(index="date", columns=["store_nbr", "family"], values=col).reindex(
            index=dates, columns=groups
        ).to_numpy(dtype=np.float32)

    sales = matrix(train, "sales")
    promo = matrix(pd.concat([train.drop(columns="sales"), test]), "onpromotion")
    stores = pd.read_csv(directory / "stores.csv")
    if stores.store_nbr.duplicated().any():
        raise ValueError("stores.csv must contain one row per store")
    meta = groups.to_frame(index=False).merge(stores, on="store_nbr", validate="many_to_one")
    if meta.isna().any().any():
        raise ValueError("Missing store metadata")
    names = [x for x in ["store_nbr", "family", "city", "state", "type", "cluster"] if x in meta]
    static = np.column_stack([pd.factorize(meta[x], sort=True)[0] for x in names]).astype("float32")
    n_train_days = (train.date.max() - train.date.min()).days + 1
    quality = {
        "train_rows": len(train), "test_rows": len(test), "series": len(groups),
        "train_start": str(train.date.min().date()), "train_end": str(train.date.max().date()),
        "test_start": str(test.date.min().date()), "test_end": str(test.date.max().date()),
        "missing_train_grid_cells": int(np.isnan(sales[:n_train_days]).sum()),
        "missing_promotion_grid_cells": int(np.isnan(promo).sum()),
        "zero_sales_fraction_observed": float((train.sales == 0).mean()),
        "policy": "Missing history: 0 plus observed flag; missing targets excluded, not imputed.",
        "auxiliary_policy": "Starter uses stores/promotion/calendar. Holiday/oil/transactions pending experiments."
    }
    return Panel(dates, groups, sales, promo, static, names,
                 pd.factorize(meta.store_nbr, sort=True)[0],
                 pd.factorize(meta.family, sort=True)[0], train, test, quality)
