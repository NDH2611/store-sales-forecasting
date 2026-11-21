"""Optional dependencies are imported only for the selected model."""

from pathlib import Path
import numpy as np
import pandas as pd

from .features import origin_dates, tabular, window
from .preprocessing import HOLIDAY_FLAGS, build_holiday_features, build_oil_features

OIL_FEATURE_COLS = [
    "oil_price",
    "oil_unobserved_at_origin",
    "oil_still_missing",
    "oil_is_forecast",
]


def _resolve_data_dir(config) -> Path:
    if "_data_dir" in config and config["_data_dir"]:
        return Path(config["_data_dir"])
    default_dir = Path("data")
    if default_dir.exists():
        return default_dir
    raise ValueError(
        "Missing '_data_dir' in config and default 'data/' directory not found. "
        "Pass config['_data_dir'] when calling fit_lightgbm directly."
    )


def _build_aux_context(panel, config):
    fset = config["feature_set"]
    use_holiday = fset in ("holiday", "holiday_oil")
    use_oil = fset in ("oil", "holiday_oil")
    if not (use_holiday or use_oil):
        return None

    data_dir = _resolve_data_dir(config)
    start_date = panel.dates[0]
    # Đảm bảo end_date luôn phủ hết cả horizon của tập test
    end_date = max(
        pd.Timestamp(panel.dates[-1]),
        pd.Timestamp(panel.dates[-1]) + pd.Timedelta(days=config["horizon"]),
    )

    ctx = {
        "store_nbrs": np.array([g[0] for g in panel.groups], dtype="int64"),
        "holiday_lookup": None,
        "oil_df": None,
    }

    if use_holiday:
        holidays = pd.read_csv(data_dir / "holidays_events.csv")
        stores = pd.read_csv(data_dir / "stores.csv")
        hol_df = build_holiday_features(holidays, stores, start_date, end_date)
        ctx["holiday_lookup"] = hol_df.set_index(["date", "store_nbr"])[HOLIDAY_FLAGS]

    if use_oil:
        ctx["oil_df"] = pd.read_csv(data_dir / "oil.csv")

    return ctx


def _window_aux(panel, origin, horizon, ctx):
    if ctx is None:
        return None

    t = panel.position(origin)
    future_dates = panel.dates[t + 1 : t + horizon + 1]
    groups = len(panel.groups)
    aux = {}

    if ctx["holiday_lookup"] is not None:
        idx = pd.MultiIndex.from_arrays(
            [
                np.tile(future_dates, groups),
                np.repeat(ctx["store_nbrs"], horizon),
            ],
            names=["date", "store_nbr"],
        )
        hol_vals = ctx["holiday_lookup"].reindex(idx).fillna(0.0).to_numpy(dtype="float32")
        for i, col in enumerate(HOLIDAY_FLAGS):
            aux[col] = hol_vals[:, i]

    if ctx["oil_df"] is not None:
        oil_feat = build_oil_features(
            ctx["oil_df"], future_dates[0], future_dates[-1], cutoff=origin
        )
        for col in OIL_FEATURE_COLS:
            aux[col] = np.tile(oil_feat[col].to_numpy(dtype="float32"), groups)

    return aux


def fit_lightgbm(panel, cutoff, config, output):
    import lightgbm as lgb
    from lightgbm import LGBMRegressor

    aux_ctx = _build_aux_context(panel, config)
    origins = list(origin_dates(panel, cutoff, config))
    es_rounds = config.get("early_stopping_rounds", None)

    # Nếu bật early_stopping_rounds và có đủ nhiều origin, tách origin cuối cùng (kết thúc tại cutoff) làm val nội bộ
    use_es = bool(es_rounds and len(origins) > 16)
    val_origin = origins[-1] if use_es else None
    # Loại bỏ các origin có target chồng lấn với val_origin (cách val_origin < horizon ngày) để chống rò rỉ
    train_origins = (
        [
            o
            for o in origins[:-1]
            if (pd.Timestamp(val_origin) - pd.Timestamp(o)).days >= config["horizon"]
        ]
        if use_es
        else origins
    )

    frames, targets = [], []
    for origin in train_origins:
        hist, future, target = window(panel, origin, config["lookback"], config["horizon"])
        aux = _window_aux(panel, origin, config["horizon"], aux_ctx)
        frame = tabular(panel, hist, future, config["feature_set"], aux=aux)
        mask = np.isfinite(target.reshape(-1))
        frames.append(frame.loc[mask])
        targets.append(np.log1p(target.reshape(-1)[mask]))

    X_train = pd.concat(frames, ignore_index=True)
    y_train = np.concatenate(targets)

    lgb_cfg = dict(config["lightgbm"])
    if lgb_cfg.get("subsample", 1.0) < 1.0 and "subsample_freq" not in lgb_cfg:
        lgb_cfg["subsample_freq"] = 1

    model = LGBMRegressor(random_state=config["seed"], verbosity=-1, **lgb_cfg)

    if use_es:
        v_hist, v_future, v_target = window(
            panel, val_origin, config["lookback"], config["horizon"]
        )
        v_aux = _window_aux(panel, val_origin, config["horizon"], aux_ctx)
        v_frame = tabular(panel, v_hist, v_future, config["feature_set"], aux=v_aux)
        v_mask = np.isfinite(v_target.reshape(-1))
        X_val = v_frame.loc[v_mask]
        y_val = np.log1p(v_target.reshape(-1)[v_mask])

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=panel.static_names,
            callbacks=[lgb.early_stopping(int(es_rounds), verbose=False)],
        )
    else:
        model.fit(X_train, y_train, categorical_feature=panel.static_names)

    model.booster_.save_model(str(output / "lightgbm.txt"))
    hist, future, _ = window(panel, cutoff, config["lookback"], config["horizon"])
    aux = _window_aux(panel, cutoff, config["horizon"], aux_ctx)
    frame = tabular(panel, hist, future, config["feature_set"], aux=aux)
    prediction = model.predict(frame).reshape(len(panel.groups), config["horizon"])
    sales_pred = np.maximum(0, np.expm1(prediction))
    zero_21d = (hist[:, -min(21, hist.shape[1]) :, 0].sum(axis=1) == 0)[:, None]
    return np.where(zero_21d, 0.0, sales_pred)


def make_lstm(panel, config):
    import torch
    from torch import nn

    class Forecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.store = nn.Embedding(int(panel.store_codes.max()) + 1, 8)
            self.family = nn.Embedding(int(panel.family_codes.max()) + 1, 8)
            hidden = config["lstm"]["hidden_size"]
            self.encoder = nn.LSTM(3, hidden, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden + 16 + 6 * config["horizon"], hidden),
                nn.ReLU(),
                nn.Linear(hidden, config["horizon"]),
            )

        def forward(self, history, future, store, family):
            _, (hidden, _) = self.encoder(history)
            inputs = torch.cat(
                [hidden[-1], self.store(store), self.family(family), future.flatten(1)],
                dim=1,
            )
            return self.head(inputs)

    return Forecaster()


def fit_lstm(panel, cutoff, config, output):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    histories, futures, targets, stores, families = [], [], [], [], []
    for origin in origin_dates(panel, cutoff, config):
        hist, future, target = window(panel, origin, config["lookback"], config["horizon"])
        valid = np.isfinite(target).all(axis=1)
        histories.append(hist[valid])
        futures.append(future[valid])
        targets.append(np.log1p(target[valid]))
        stores.append(panel.store_codes[valid])
        families.append(panel.family_codes[valid])
    arrays = [np.concatenate(x) for x in [histories, futures, stores, families, targets]]
    if not len(arrays[0]):
        raise ValueError("No complete target windows for LSTM")
    tensors = [torch.from_numpy(x) for x in arrays]
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=config["lstm"]["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_lstm(panel, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lstm"]["learning_rate"])
    curve = []
    for epoch in range(config["lstm"]["epochs"]):
        model.train()
        total = 0.0
        for batch in loader:
            history, future, store, family, y = [x.to(device) for x in batch]
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(history, future, store, family), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(y)
        curve.append({"epoch": epoch + 1, "train_mse_log": total / len(arrays[0])})
        print(curve[-1], flush=True)
    pd.DataFrame(curve).to_csv(output / "learning_curve.csv", index=False)
    torch.save(model.cpu().state_dict(), output / "lstm.pt")
    model.eval()
    hist, future, _ = window(panel, cutoff, config["lookback"], config["horizon"])
    with torch.no_grad():
        predicted = model(
            torch.from_numpy(hist),
            torch.from_numpy(future),
            torch.from_numpy(panel.store_codes),
            torch.from_numpy(panel.family_codes),
        )
    return np.maximum(0, np.expm1(predicted.numpy()))
