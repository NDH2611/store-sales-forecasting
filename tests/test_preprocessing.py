import unittest

import pandas as pd

from store_sales.preprocessing import (
    build_holiday_features,
    build_oil_features,
)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        # Dữ liệu giả lập nhỏ để kiểm tra logic.
        self.stores = pd.DataFrame({
            "store_nbr": [1, 2, 3],
            "city": ["A", "B", "C"],
            "state": ["X", "X", "Y"],
        })

        self.holidays = pd.DataFrame(
            [
                ["2017-01-01", "Holiday", "National", "Ecuador", True],
                ["2017-01-02", "Transfer", "National", "Ecuador", False],
                ["2017-01-03", "Holiday", "Local", "A", False],
                ["2017-01-03", "Additional", "Local", "A", False],
                ["2017-01-04", "Bridge", "Regional", "X", False],
            ],
            columns=[
                "date", "type", "locale",
                "locale_name", "transferred",
            ],
        )

        self.oil = pd.DataFrame({
            "date": [
                "2017-01-01",
                "2017-01-02",
                "2017-01-04",
            ],
            "dcoilwtico": [None, 50.0, 999.0],
        })

    def test_holiday_scope_transfer_and_unique_keys(self):
        result = build_holiday_features(
            self.holidays,
            self.stores,
            "2017-01-01",
            "2017-01-04",
        )

        self.assertEqual(len(result), 12)
        self.assertFalse(
            result.duplicated(["date", "store_nbr"]).any()
        )

        totals = result.groupby("date").sum(numeric_only=True)

        # Ngày gốc bị chuyển: không tính là ngày nghỉ từ sự kiện đó.
        self.assertEqual(
            totals.loc["2017-01-01", "is_holiday"], 0
        )
        self.assertEqual(
            totals.loc["2017-01-01", "is_transferred_original"], 3
        )

        # Ngày chuyển tới áp dụng cho cả 3 cửa hàng.
        self.assertEqual(
            totals.loc["2017-01-02", "is_transfer"], 3
        )

        # Hai sự kiện Local chỉ áp dụng cửa hàng ở thành phố A.
        self.assertEqual(
            totals.loc["2017-01-03", "is_holiday"], 1
        )
        self.assertEqual(
            totals.loc["2017-01-03", "is_additional"], 1
        )

        # Regional X áp dụng cho 2 cửa hàng.
        self.assertEqual(
            totals.loc["2017-01-04", "is_bridge"], 2
        )

    def test_no_events_returns_zero_flags(self):
        result = build_holiday_features(
            self.holidays,
            self.stores,
            "2017-02-01",
            "2017-02-02",
        )

        self.assertEqual(len(result), 6)
        self.assertEqual(result["is_holiday"].sum(), 0)

    def test_unknown_locality_rejected(self):
        self.holidays.loc[2, "locale_name"] = "Unknown"

        with self.assertRaises(ValueError):
            build_holiday_features(
                self.holidays,
                self.stores,
                "2017-01-01",
                "2017-01-04",
            )

    def test_duplicate_store_rejected(self):
        duplicate_stores = pd.concat([
            self.stores,
            self.stores.iloc[:1],
        ])

        with self.assertRaises(ValueError):
            build_holiday_features(
                self.holidays,
                duplicate_stores,
                "2017-01-01",
                "2017-01-04",
            )

    def test_oil_past_only_and_initial_missing(self):
        result = build_oil_features(
            self.oil,
            "2017-01-01",
            "2017-01-05",
            "2017-01-02",
        )

        self.assertTrue(pd.isna(result.loc[0, "oil_price"]))

        self.assertEqual(
            result.loc[1:, "oil_price"].tolist(),
            [50.0] * 4,
        )

        self.assertEqual(
            result["oil_is_forecast"].tolist(),
            [0, 0, 1, 1, 1],
        )

        self.assertEqual(
            result["oil_unobserved_at_origin"].tolist(),
            [1, 0, 1, 1, 1],
        )

    def test_future_oil_cannot_change_features(self):
        before = build_oil_features(
            self.oil,
            "2017-01-01",
            "2017-01-05",
            "2017-01-02",
        )

        # Sửa giá SAU cutoff không được làm thay đổi đầu vào.
        self.oil.loc[2, "dcoilwtico"] = -12345

        after = build_oil_features(
            self.oil,
            "2017-01-01",
            "2017-01-05",
            "2017-01-02",
        )

        pd.testing.assert_frame_equal(before, after)

    def test_oil_carries_price_from_before_start(self):
        result = build_oil_features(
            self.oil,
            "2017-01-03",
            "2017-01-05",
            "2017-01-02",
        )

        self.assertEqual(
            result["oil_price"].tolist(),
            [50.0] * 3,
        )

    def test_duplicate_oil_date_rejected(self):
        duplicate_oil = pd.concat([
            self.oil,
            self.oil.iloc[:1],
        ])

        with self.assertRaises(ValueError):
            build_oil_features(
                duplicate_oil,
                "2017-01-01",
                "2017-01-05",
                "2017-01-02",
            )


if __name__ == "__main__":
    unittest.main()