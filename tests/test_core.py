import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from store_sales.data import load_panel, validate_rows
from store_sales.evaluate import prediction_frame, rmsle, score, submission, validate_submission
from store_sales.features import origin_dates, seasonal_naive, tabular, window


def make_fixture(directory):
    """Synthetic TEST DATA, not competition results."""
    dates = pd.date_range("2016-01-01", periods=420)
    index = pd.MultiIndex.from_product([dates, [1, 2], ["A", "B"]],
                                      names=["date", "store_nbr", "family"])
    data = index.to_frame(index=False)
    data["id"] = np.arange(len(data))
    data["sales"] = data.date.dt.dayofweek + data.store_nbr*10.0
    data["onpromotion"] = (data.date.dt.dayofweek == 5).astype(int)
    data.iloc[:-64].to_csv(directory/"train.csv", index=False)
    data.iloc[-64:].drop(columns="sales").to_csv(directory/"test.csv", index=False)
    pd.DataFrame({"store_nbr": [1, 2], "city": ["X", "Y"], "state": ["S", "S"],
                  "type": ["A", "B"], "cluster": [1, 2]}).to_csv(directory/"stores.csv", index=False)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        make_fixture(self.path)
        self.panel = load_panel(self.path)
        self.cutoff = self.panel.dates[-33]
        self.config = {"horizon": 16, "lookback": 56, "stride": 16, "train_days": 128}

    def tearDown(self):
        self.temp.cleanup()

    def test_rmsle_known_value_and_invalid_inputs(self):
        self.assertEqual(rmsle([1, 2], [1, 2]), 0)
        self.assertAlmostEqual(rmsle([0], [1]), np.log(2))
        for true, pred in [([1], [-1]), ([1], [np.nan]), ([1, 2], [1]), ([], [])]:
            with self.assertRaises(ValueError):
                rmsle(true, pred)

    def test_future_targets_cannot_change_features_or_baseline(self):
        before_h, before_f, _ = window(self.panel, self.cutoff, 56, 16)
        before_x = tabular(self.panel, before_h, before_f)
        before_base = seasonal_naive(self.panel, self.cutoff, 16)
        self.panel.sales[self.panel.position(self.cutoff)+1:] = 999999
        after_h, after_f, _ = window(self.panel, self.cutoff, 56, 16)
        np.testing.assert_array_equal(before_h, after_h)
        np.testing.assert_array_equal(before_f, after_f)
        pd.testing.assert_frame_equal(before_x, tabular(self.panel, after_h, after_f))
        np.testing.assert_array_equal(before_base, seasonal_naive(self.panel, self.cutoff, 16))

    def test_training_targets_end_before_forecast(self):
        origins = origin_dates(self.panel, self.cutoff, self.config)
        for date in origins:
            self.assertLessEqual(date+pd.Timedelta(days=16), self.cutoff)

    def test_missing_history_is_flagged_not_hidden(self):
        t = self.panel.position(self.cutoff)
        self.panel.sales[t, 0] = np.nan
        hist, _, _ = window(self.panel, self.cutoff, 56, 16)
        self.assertEqual(hist[0, -1, 0], 0)
        self.assertEqual(hist[0, -1, 2], 0)

    def test_missing_future_promotion_fails(self):
        self.panel.promo[self.panel.position(self.cutoff)+1, 0] = np.nan
        with self.assertRaises(ValueError):
            window(self.panel, self.cutoff, 56, 16)

    def test_training_skips_origins_with_unknown_future_promotion(self):
        origins = origin_dates(self.panel, self.cutoff, self.config)
        first = origins[0]
        self.panel.promo[self.panel.position(first)+1, 0] = np.nan
        kept = origin_dates(self.panel, self.cutoff, self.config)
        self.assertNotIn(first, kept)
        self.assertLess(len(kept), len(origins))

    def test_baseline_recovers_weekly_pattern_and_score(self):
        prediction = seasonal_naive(self.panel, self.cutoff, 16)
        result = score(self.panel, self.cutoff, prediction, self.path)
        self.assertAlmostEqual(result["rmsle"], 0)
        self.assertEqual(result["n"], 64)

    def test_score_excludes_missing_validation_targets_without_imputing_zero(self):
        prediction = seasonal_naive(self.panel, self.cutoff, 16)
        validation_date = pd.Timestamp(self.cutoff) + pd.Timedelta(days=1)
        missing_key = ((self.panel.train.date == validation_date) &
                       (self.panel.train.store_nbr == 1) &
                       (self.panel.train.family == "A"))
        self.panel.train = self.panel.train.loc[~missing_key].copy()
        result = score(self.panel, self.cutoff, prediction, self.path)
        self.assertEqual(result["n"], 63)
        self.assertEqual(result["n_forecast_expected"], 64)
        self.assertEqual(result["n_actual_observed"], 63)
        self.assertEqual(result["n_target_missing"], 1)
        self.assertAlmostEqual(result["target_coverage_pct"], 100 * 63 / 64)
        scored = pd.read_csv(self.path / "validation_predictions.csv")
        self.assertEqual(len(scored), 63)
        self.assertFalse(((scored.date == str(validation_date.date())) &
                          (scored.store_nbr == 1) & (scored.family == "A")).any())

    def test_score_errors_when_an_observed_actual_has_no_prediction(self):
        from unittest.mock import patch

        prediction = seasonal_naive(self.panel, self.cutoff, 16)
        incomplete = prediction_frame(self.panel, self.cutoff, prediction).iloc[1:].copy()
        with patch("store_sales.evaluate.prediction_frame", return_value=incomplete):
            with self.assertRaisesRegex(ValueError, "missing keys with observed actuals"):
                score(self.panel, self.cutoff, prediction, self.path)

    def test_score_key_alignment_preserves_actual_order(self):
        from unittest.mock import patch

        prediction = seasonal_naive(self.panel, self.cutoff, 16)
        shuffled = prediction_frame(self.panel, self.cutoff, prediction).sample(
            frac=1, random_state=7
        )
        with patch("store_sales.evaluate.prediction_frame", return_value=shuffled):
            result = score(self.panel, self.cutoff, prediction, self.path)
        expected_keys = self.panel.train.loc[
            (self.panel.train.date > pd.Timestamp(self.cutoff)) &
            (self.panel.train.date <= pd.Timestamp(self.cutoff) + pd.Timedelta(days=16)),
            ["date", "store_nbr", "family"],
        ].reset_index(drop=True)
        scored_keys = pd.read_csv(self.path / "validation_predictions.csv",
                                  parse_dates=["date"])[
                                      ["date", "store_nbr", "family"]
                                  ].reset_index(drop=True)
        pd.testing.assert_frame_equal(scored_keys, expected_keys)
        self.assertAlmostEqual(result["rmsle"], 0)

    def test_submission_checks_and_alignment(self):
        cutoff = self.panel.train.date.max()
        pred = seasonal_naive(self.panel, cutoff, 16)
        result = submission(self.panel, cutoff, pred, self.path)
        validate_submission(result, self.panel.test)
        with self.assertRaises(ValueError):
            validate_submission(result.iloc[::-1], self.panel.test)
        result.loc[0, "sales"] = np.inf
        with self.assertRaises(ValueError):
            validate_submission(result, self.panel.test)

    def test_duplicate_and_bad_join_rejected(self):
        with self.assertRaises(ValueError):
            validate_rows(pd.concat([self.panel.train, self.panel.train.iloc[:1]]), True)
        stores = pd.read_csv(self.path/"stores.csv")
        pd.concat([stores, stores.iloc[:1]]).to_csv(self.path/"stores.csv", index=False)
        with self.assertRaises(ValueError):
            load_panel(self.path)

    def test_cli_baseline_writes_auditable_validation_run(self):
        import subprocess
        import sys
        config = self.path/"config.json"
        config.write_text(json.dumps({**self.config, "seed": 42, "feature_set": "full"}))
        out = self.path/"run"
        subprocess.run([sys.executable, "-m", "store_sales.run", "--data", str(self.path),
                        "--config", str(config), "--model", "baseline", "--cutoff",
                        str(self.cutoff.date()), "--out", str(out)], check=True,
                       capture_output=True, text=True)
        info = json.loads((out/"run.json").read_text())
        self.assertAlmostEqual(info["metrics"]["rmsle"], 0)
        self.assertEqual(len(info["data_sha256"]), 3)


if __name__ == "__main__":
    unittest.main()
