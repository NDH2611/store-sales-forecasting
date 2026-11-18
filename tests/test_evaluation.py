import unittest

import numpy as np
import pandas as pd

from store_sales.evaluate import (
    compare_models,
    error_analysis,
    evaluate_predictions,
    rmsle,
    validate_predictions,
)


def controlled_rows():
    actual = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"]),
        "store_nbr": [1, 1, 1, 1],
        "family": ["A", "B", "A", "B"],
        "sales": [0.0, 1.0, 3.0, 7.0],
        "id": [10, 11, 12, 13],
    })
    predictions = actual[["date", "store_nbr", "family", "id"]].copy()
    predictions["prediction"] = [0.0, 3.0, 3.0, 3.0]
    return actual, predictions


class EvaluationTests(unittest.TestCase):
    def test_rmsle_known_value(self):
        self.assertAlmostEqual(rmsle([0, 1], [0, 3]), np.log(2) / np.sqrt(2))

    def test_valid_forecast_and_group_metrics(self):
        actual, predictions = controlled_rows()
        result = evaluate_predictions(actual, predictions, model="fixture", fold="f1")
        self.assertEqual(result["metrics"]["n"], 4)
        self.assertEqual(result["metrics"]["model"], "fixture")
        self.assertEqual(set(result["slices"].slice), {"family", "store_nbr", "horizon"})
        self.assertEqual(len(result["slices"].query("slice == 'horizon'")), 2)
        self.assertEqual(error_analysis(result, worst_n=1).iloc[0].family, "B")
        self.assertEqual(compare_models({"fixture": result}).iloc[0].model, "fixture")

    def test_invalid_prediction_rows_are_rejected(self):
        actual, predictions = controlled_rows()
        mutations = []
        duplicate = predictions.copy()
        duplicate.loc[1, "date"] = duplicate.loc[0, "date"]
        duplicate.loc[1, "family"] = duplicate.loc[0, "family"]
        mutations.append(duplicate)
        mutations.append(predictions.iloc[:-1].copy())
        nan = predictions.copy()
        nan.loc[0, "prediction"] = np.nan
        mutations.append(nan)
        negative = predictions.copy()
        negative.loc[0, "prediction"] = -1
        mutations.append(negative)
        wrong_id = predictions.copy()
        wrong_id.loc[0, "id"] = 99
        mutations.append(wrong_id)
        for bad in mutations:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_predictions(actual, bad)

    def test_order_and_date_range_are_checked(self):
        actual, predictions = controlled_rows()
        with self.assertRaisesRegex(ValueError, "keys/order"):
            validate_predictions(actual, predictions.iloc[::-1].reset_index(drop=True))
        gap = predictions.copy()
        gap.loc[2:, "date"] = pd.Timestamp("2020-01-04")
        gap.loc[2:, "id"] = [12, 13]
        with self.assertRaises(ValueError):
            validate_predictions(actual, gap)
        with self.assertRaisesRegex(ValueError, "origin/horizon"):
            validate_predictions(actual, predictions, origin="2020-01-02")

    def test_missing_id_and_actual_values_fail(self):
        actual, predictions = controlled_rows()
        with self.assertRaisesRegex(ValueError, "missing IDs"):
            validate_predictions(actual, predictions.drop(columns="id"), expected_ids=True)
        bad_actual = actual.copy()
        bad_actual.loc[0, "sales"] = np.inf
        with self.assertRaises(ValueError):
            validate_predictions(bad_actual, predictions)


if __name__ == "__main__":
    unittest.main()
