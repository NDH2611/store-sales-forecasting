# ALL CONFIGS

## 1. abl_feat_full.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "full",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 2. abl_feat_holiday_oil.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday_oil",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 3. abl_feat_oil.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "oil",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 4. abl_win_365d.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 365,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 5. abl_win_400d.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 400,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 6. abl_win_730d.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 730,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 7. ablation_v1.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 730,
    "stride": 7,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample_for_bin": 50000,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "n_jobs": -1
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    }
}
`

## 8. ablation_v2_oil.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 730,
    "stride": 7,
    "feature_set": "holiday_oil",
    "lightgbm": {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample_for_bin": 50000,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "n_jobs": -1
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    }
}
`

## 9. ablation_v3_450d.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 7,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 700,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample_for_bin": 50000,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "n_jobs": -1
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    }
}
`

## 10. boost_holiday.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 400,
    "stride": 7,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 1200,
        "learning_rate": 0.025,
        "num_leaves": 127,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "n_jobs": -1
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    }
}
`

## 11. boost_v1.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 400,
    "stride": 7,
    "feature_set": "full",
    "lightgbm": {
        "n_estimators": 1500,
        "learning_rate": 0.025,
        "num_leaves": 127,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "n_jobs": -1
    }
}
`

## 12. default.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 365,
    "stride": 16,
    "feature_set": "full",
    "lightgbm": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    }
}
`

## 13. full_tuned.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 730,
    "stride": 7,
    "feature_set": "full",
    "lightgbm": {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample_for_bin": 50000,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "n_jobs": -1
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    }
}
`

## 14. holiday.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 365,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    }
}
`

## 15. holiday_oil.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 365,
    "stride": 16,
    "feature_set": "holiday_oil",
    "lightgbm": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    }
}
`

## 16. holiday_smooth.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 730,
    "stride": 7,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 1200,
        "learning_rate": 0.015,
        "num_leaves": 31,
        "subsample_for_bin": 50000,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "min_child_samples": 50,
        "n_jobs": -1
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    }
}
`

## 17. holiday_tuned.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 730,
    "stride": 7,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample_for_bin": 50000,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "n_jobs": -1
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    }
}
`

## 18. lgbm_no_history.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 365,
    "stride": 16,
    "feature_set": "no_history",
    "lightgbm": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    }
}
`

## 19. lgbm_no_promo.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 365,
    "stride": 16,
    "feature_set": "no_promo",
    "lightgbm": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    }
}
`

## 20. oil.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 365,
    "stride": 16,
    "feature_set": "oil",
    "lightgbm": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    }
}
`

## 21. opt_730_hol_1200.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 730,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": null
}
`

## 22. opt_730_hol_1600_lr02.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 730,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 1600,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": null
}
`

## 23. opt_730_hol_900.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 730,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 900,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": null
}
`

## 24. opt_730_holoil_1200.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 730,
    "stride": 16,
    "feature_set": "holiday_oil",
    "lightgbm": {
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": null
}
`

## 25. opt_730_holoil_1600_lr02.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 730,
    "stride": 16,
    "feature_set": "holiday_oil",
    "lightgbm": {
        "n_estimators": 1600,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": null
}
`

## 26. opt_730_holoil_l63_mcs35_reg.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 730,
    "stride": 16,
    "feature_set": "holiday_oil",
    "lightgbm": {
        "n_estimators": 1400,
        "learning_rate": 0.025,
        "num_leaves": 63,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 35,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": null
}
`

## 27. opt_730_holoil_l63_samp0806.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 730,
    "stride": 16,
    "feature_set": "holiday_oil",
    "lightgbm": {
        "n_estimators": 1200,
        "learning_rate": 0.025,
        "num_leaves": 63,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": null
}
`

## 28. opt_730_holoil_samp0806.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 730,
    "stride": 16,
    "feature_set": "holiday_oil",
    "lightgbm": {
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": null
}
`

## 29. opt_730_l63_mcs35_reg.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 730,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 1400,
        "learning_rate": 0.025,
        "num_leaves": 63,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 35,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": null
}
`

## 30. opt_730_l63_samp0806.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 730,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 1200,
        "learning_rate": 0.025,
        "num_leaves": 63,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": null
}
`

## 31. opt_730_samp0806.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 730,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": null
}
`

## 32. opt_800_holoil_best.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 800,
    "stride": 16,
    "feature_set": "holiday_oil",
    "lightgbm": {
        "n_estimators": 1400,
        "learning_rate": 0.025,
        "num_leaves": 63,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 35,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": null
}
`

## 33. ref_ablation_450d.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 700,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": null
}
`

## 34. ref_boost_400d.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 400,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 1200,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": null
}
`

## 35. ref_default_365d.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 365,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": null
}
`

## 36. smoke.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 28,
    "train_days": 64,
    "stride": 16,
    "feature_set": "full",
    "lightgbm": {
        "n_estimators": 10,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "n_jobs": 2
    },
    "lstm": {
        "hidden_size": 16,
        "epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001
    }
}
`

## 37. tune_base_es50.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 38. tune_d5_l25.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 25,
        "n_jobs": 2,
        "max_depth": 5,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 39. tune_d6_l45.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 45,
        "n_jobs": 2,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 40. tune_d7_l63.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "n_jobs": 2,
        "max_depth": 7,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 41. tune_d8_l95.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 95,
        "n_jobs": 2,
        "max_depth": 8,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 42. tune_dinf_l63.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 43. tune_lr_0015.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.015,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 44. tune_lr_0030.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 45. tune_lr_0080.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.08,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 46. tune_mcs_100.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 100,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 47. tune_mcs_150.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 150,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 48. tune_mcs_50.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 49. tune_reg_01_10.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 50. tune_reg_10_50.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 51. tune_reg_50_100.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 1,
        "colsample_bytree": 1.0,
        "reg_alpha": 5.0,
        "reg_lambda": 10.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 52. tune_samp_07_07.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 53. tune_samp_08_06.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

## 54. tune_samp_08_08.json

`json
{
    "seed": 42,
    "horizon": 16,
    "lookback": 56,
    "train_days": 450,
    "stride": 16,
    "feature_set": "holiday",
    "lightgbm": {
        "n_estimators": 2500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 2,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0
    },
    "lstm": {
        "hidden_size": 64,
        "epochs": 10,
        "batch_size": 512,
        "learning_rate": 0.001
    },
    "early_stopping_rounds": 50
}
`

