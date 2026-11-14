"""XAI for a saved validation run, never fabricated training results."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_panel
from .features import tabular, window


def explain_tree(panel, run, config, cutoff):
    import lightgbm as lgb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    hist, future, _ = window(panel, cutoff, config["lookback"], config["horizon"])
    frame = tabular(panel, hist, future, config["feature_set"])
    ids = np.random.default_rng(config["seed"]).choice(len(frame), min(512, len(frame)), replace=False)
    sample = frame.iloc[ids]
    model = lgb.Booster(model_file=str(run/"lightgbm.txt"))
    values = shap.TreeExplainer(model)(sample)
    pd.DataFrame({"feature": frame.columns, "mean_abs_shap": np.abs(values.values).mean(axis=0)}).sort_values(
        "mean_abs_shap", ascending=False).to_csv(run/"xai_global.csv", index=False)
    key = panel.groups.to_frame(index=False).iloc[ids//config["horizon"]].reset_index(drop=True)
    key["horizon"] = ids % config["horizon"] + 1
    pd.concat([key, pd.DataFrame(values.values, columns=frame.columns)], axis=1).to_csv(
        run/"xai_shap_values.csv", index=False)
    shap.plots.beeswarm(values, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(run/"xai_global.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    for i in range(min(3, len(sample))):
        shap.plots.waterfall(values[i], show=False)
        plt.savefig(run/f"xai_example_{i}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    (run/"xai_notes.txt").write_text(
        "SHAP explains raw model log1p(sales), before expm1 and nonnegative clipping. "
        "Sampled validation rows; three random examples are a starter. "
        "For report add typical/promotion/large-error cases with keys. "
        "Associations explain this model, not causal effects.", encoding="utf-8")


def explain_neural(panel, run, config, cutoff):
    import torch
    from .models import make_lstm

    model = make_lstm(panel, config)
    model.load_state_dict(torch.load(run/"lstm.pt", map_location="cpu", weights_only=True))
    model.eval()
    hist, future, _ = window(panel, cutoff, config["lookback"], config["horizon"])
    selected = np.random.default_rng(config["seed"]).choice(len(hist), min(24, len(hist)), replace=False)
    records, checks = [], []
    # Conditional IG: explain continuous sales/promo history and future promo.
    # Calendar, observed flags, store/family embeddings remain fixed at actual context.
    for group in selected:
        actual_h = torch.from_numpy(hist[group:group+1])
        actual_f = torch.from_numpy(future[group:group+1])
        base_h, base_f = actual_h.clone(), actual_f.clone()
        base_h[:, :, :2] = 0
        base_f[:, :, 5] = 0
        store = torch.tensor(panel.store_codes[group:group+1])
        family = torch.tensor(panel.family_codes[group:group+1])
        for horizon_index in [0, 7, 15]:
            sum_h, sum_f = torch.zeros_like(actual_h), torch.zeros_like(actual_f)
            # Midpoint quadrature; completeness residual is recorded explicitly.
            for step in range(32):
                alpha = (step+0.5)/32
                path_h = (base_h+alpha*(actual_h-base_h)).requires_grad_(True)
                path_f = (base_f+alpha*(actual_f-base_f)).requires_grad_(True)
                output = model(path_h, path_f, store, family)[0, horizon_index]
                grad_h, grad_f = torch.autograd.grad(output, (path_h, path_f))
                sum_h += grad_h.detach()/32
                sum_f += grad_f.detach()/32
            attr_h = ((actual_h-base_h)*sum_h).numpy()[0]
            attr_f = ((actual_f-base_f)*sum_f).numpy()[0]
            with torch.no_grad():
                difference = float(model(actual_h, actual_f, store, family)[0, horizon_index] -
                                   model(base_h, base_f, store, family)[0, horizon_index])
            checks.append({"group": int(group), "horizon": horizon_index+1,
                           "output_difference_log": difference,
                           "attribution_sum": float(attr_h.sum()+attr_f.sum()),
                           "completeness_residual": difference-float(attr_h.sum()+attr_f.sum())})
            for index in range(len(attr_h)):
                for channel, name in enumerate(["history_sales_log", "history_promo_log"]):
                    records.append({"store_nbr": panel.groups[group][0], "family": panel.groups[group][1],
                                    "horizon": horizon_index+1, "feature": name,
                                    "relative_day": index-len(attr_h)+1,
                                    "attribution": float(attr_h[index, channel])})
            for index in range(len(attr_f)):
                records.append({"store_nbr": panel.groups[group][0], "family": panel.groups[group][1],
                                "horizon": horizon_index+1, "feature": "future_promo_log",
                                "relative_day": index+1, "attribution": float(attr_f[index, 5])})
    pd.DataFrame(records).to_csv(run/"xai_integrated_gradients.csv", index=False)
    pd.DataFrame(checks).to_csv(run/"xai_completeness.csv", index=False)
    (run/"xai_notes.txt").write_text(
        "Conditional Integrated Gradients in raw log1p(sales) output. "
        "Zero sales/promo baseline; calendar, observation flags and identities held fixed. "
        "24 sampled series, horizons 1/8/16, 32 integration steps. "
        "Check completeness residual; repeat with alternative baseline and more steps. "
        "This does not explain categorical embeddings and is not causal evidence.", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    info = json.loads((args.run/"run.json").read_text())
    if not (args.run/"validation_predictions.csv").exists():
        parser.error("Explain a saved validation run; select report cases with known outcomes")
    panel = load_panel(args.data)
    if info["model"] == "lightgbm":
        explain_tree(panel, args.run, info["config"], info["cutoff"])
    elif info["model"] == "lstm":
        explain_neural(panel, args.run, info["config"], info["cutoff"])
    else:
        parser.error("XAI starter supports lightgbm/lstm; explain baseline explicitly if it wins")


if __name__ == "__main__":
    main()
