"""python -m store_sales.run --help"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_panel
from .evaluate import prediction_frame, score, submission
from .features import seasonal_naive


def main():
    parser = argparse.ArgumentParser(description="Reproducible origin-based 16-day forecasting")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--config", default="configs/default.json", type=Path)
    parser.add_argument("--model", choices=["baseline", "lightgbm", "lstm"], default="baseline")
    parser.add_argument("--cutoff", help="Last observed date; omit for final train/test submission")
    parser.add_argument("--out", required=True, type=Path, help="A NEW directory for this run")
    parser.add_argument("--allow-holdout", action="store_true", help="Explicit one-time final internal evaluation")
    args = parser.parse_args()
    if args.cutoff and pd.Timestamp(args.cutoff) >= pd.Timestamp("2017-07-30") and not args.allow_holdout:
        parser.error("Frozen holdout: use --allow-holdout only AFTER freezing model/config choices")
    config = json.loads(args.config.read_text())
    if config["horizon"] != 16 or config["lookback"] < 28 or config["stride"] < 1:
        raise ValueError("Expected horizon=16, lookback>=28, stride>=1")
    if config["feature_set"] not in {"full", "no_history", "no_promo"}:
        raise ValueError("Unknown feature_set")
    if args.model == "lstm" and config["feature_set"] != "full":
        raise ValueError("LSTM starter only supports full features; implement neural ablation explicitly")
    args.out.mkdir(parents=True, exist_ok=False)
    np.random.seed(config["seed"])
    started = time.perf_counter()
    panel = load_panel(args.data)
    cutoff = args.cutoff or str(panel.train.date.max().date())
    if pd.Timestamp(cutoff) > panel.train.date.max():
        raise ValueError("Cutoff after last observed training date")
    if args.cutoff and pd.Timestamp(cutoff)+pd.Timedelta(days=config["horizon"]) > panel.train.date.max():
        raise ValueError("Validation horizon extends beyond observed labels")
    if args.model == "baseline":
        prediction = seasonal_naive(panel, cutoff, config["horizon"])
    else:
        from .models import fit_lightgbm, fit_lstm
        trainer = fit_lightgbm if args.model == "lightgbm" else fit_lstm
        prediction = trainer(panel, cutoff, config, args.out)
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite predictions; investigate model before submission")
    metrics = score(panel, cutoff, prediction, args.out) if args.cutoff else {}
    if not args.cutoff:
        submission(panel, cutoff, prediction, args.out)
    prediction_frame(panel, cutoff, prediction).to_csv(args.out / "predictions.csv", index=False)
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                          stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        git_sha, dirty = "unavailable", True
    fingerprints = {}
    for name in ["train.csv", "test.csv", "stores.csv"]:
        digest = hashlib.sha256()
        with (args.data/name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                digest.update(block)
        fingerprints[name] = digest.hexdigest()
    result = {"model": args.model, "cutoff": cutoff, "config": config, "git_sha": git_sha,
              "git_dirty": dirty, "data_sha256": fingerprints, "metrics": metrics,
              "total_seconds": time.perf_counter()-started, "python": platform.python_version(),
              "note": "Runtime includes load, fit and predict; not isolated train time."}
    provenance = Path(__file__).resolve().parents[2]/"SOURCE_PROVENANCE.json"
    if provenance.exists():
        result["source_bundle"] = json.loads(provenance.read_text())
    (args.out/"run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out/"quality.json").write_text(json.dumps(panel.quality, indent=2), encoding="utf-8")
    (args.out/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    versions = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    (args.out/"environment.txt").write_text("\n".join(versions), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
