"""Optional dependencies are imported only for the selected model."""
import numpy as np
import pandas as pd

from .features import origin_dates, tabular, window


def fit_lightgbm(panel, cutoff, config, output):
    from lightgbm import LGBMRegressor

    frames, targets = [], []
    for origin in origin_dates(panel, cutoff, config):
        hist, future, target = window(panel, origin, config["lookback"], config["horizon"])
        frame = tabular(panel, hist, future, config["feature_set"])
        mask = np.isfinite(target.reshape(-1))
        frames.append(frame.loc[mask])
        targets.append(np.log1p(target.reshape(-1)[mask]))
    features = pd.concat(frames, ignore_index=True)
    model = LGBMRegressor(random_state=config["seed"], verbosity=-1, **config["lightgbm"])
    model.fit(features, np.concatenate(targets), categorical_feature=panel.static_names)
    model.booster_.save_model(str(output / "lightgbm.txt"))
    hist, future, _ = window(panel, cutoff, config["lookback"], config["horizon"])
    frame = tabular(panel, hist, future, config["feature_set"])
    prediction = model.predict(frame).reshape(len(panel.groups), config["horizon"])
    return np.maximum(0, np.expm1(prediction))


def make_lstm(panel, config):
    import torch
    from torch import nn

    class Forecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.store = nn.Embedding(int(panel.store_codes.max())+1, 8)
            self.family = nn.Embedding(int(panel.family_codes.max())+1, 8)
            hidden = config["lstm"]["hidden_size"]
            self.encoder = nn.LSTM(3, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden+16+6*config["horizon"], hidden),
                                      nn.ReLU(), nn.Linear(hidden, config["horizon"]))

        def forward(self, history, future, store, family):
            _, (hidden, _) = self.encoder(history)
            inputs = torch.cat([hidden[-1], self.store(store), self.family(family),
                                future.flatten(1)], dim=1)
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
    loader = DataLoader(TensorDataset(*tensors), batch_size=config["lstm"]["batch_size"],
                        shuffle=True, generator=torch.Generator().manual_seed(config["seed"]))
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
            total += float(loss.detach())*len(y)
        curve.append({"epoch": epoch+1, "train_mse_log": total/len(arrays[0])})
        print(curve[-1], flush=True)
    pd.DataFrame(curve).to_csv(output / "learning_curve.csv", index=False)
    torch.save(model.cpu().state_dict(), output / "lstm.pt")
    model.eval()
    hist, future, _ = window(panel, cutoff, config["lookback"], config["horizon"])
    with torch.no_grad():
        predicted = model(torch.from_numpy(hist), torch.from_numpy(future),
                          torch.from_numpy(panel.store_codes), torch.from_numpy(panel.family_codes))
    return np.maximum(0, np.expm1(predicted.numpy()))
