# Bàn giao Dự án — Dự báo Doanh số Chuỗi thời gian (Store Sales - 100% Machine Learning)

## 1. Phạm vi công việc

Đã thực hiện:
- Xây dựng pipeline đặc trưng chuỗi thời gian nhân quả (causal tabular feature engineering) cho 1.782 chuỗi (`54 stores × 33 families`).
- Khắc phục triệt để lỗi rò rỉ dữ liệu (data leakage) và lệch chỉ mục trong ma trận khuyến mãi tương lai (`promo_log`).
- Thiết lập cơ chế kiểm thử chéo Rolling-Origin Cross-Validation 3 folds (`2017-06-12`, `2017-06-28`, `2017-07-14`), horizon 16 ngày, cố định `seed = 42`.
- Thực hiện Ablation Study đánh giá tác động của từng nhóm đặc trưng (lịch sử bán hàng, khuyến mãi, ngày lễ, giá dầu) và độ dài cửa sổ huấn luyện (`train_days`).
- Tinh chỉnh siêu tham số (Hyperparameter Tuning) cho LightGBM, chuẩn hóa kiến trúc thuần Machine Learning (loại bỏ toàn bộ thành phần Deep Learning/LSTM).
- Đóng gói cấu hình sạch, trọng số mô hình (`lightgbm.txt`), dự báo validation từng fold và nhật ký thực nghiệm vào thư mục bàn giao `project_handover/`.

Không bao gồm mô hình Deep Learning (LSTM/Neural Networks) hay dữ liệu thô (`data/`) trong gói đóng gói mã nguồn.

## 2. Thiết lập thực nghiệm & Hàm mục tiêu

- **Bài toán:** Dự báo doanh số ngày $y_{g, t+h}$ cho 1.782 chuỗi với chân trời dự báo $H = 16$ ngày (`lookback = 56`, `stride = 7`).
- **Biến đổi mục tiêu & Hàm mất mát:** Áp dụng biến đổi $z = \log(1 + \text{sales})$ trước khi huấn luyện. Mô hình tối ưu trực tiếp hàm mất mát **RMSE** trên không gian $z$, đồng nhất về mặt toán học với thước đo đánh giá **RMSLE** trên doanh số gốc:
  $$\text{RMSLE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\log(1 + \hat{y}_i) - \log(1 + y_i)\right)^2}$$
- **Tính tái lập (Reproducibility):** Cố định `seed = 42` (`random_state = 42`) cho toàn bộ quá trình lấy mẫu dòng (`subsample`), lấy mẫu cột (`colsample_bytree`), chia cây và trích mẫu giải thích mô hình.
- **Chiến lược Validation (3 Folds):**
  - **Fold 1:** `cutoff = 2017-06-12` (đánh giá `2017-06-13` $\rightarrow$ `2017-06-28`)
  - **Fold 2:** `cutoff = 2017-06-28` (đánh giá `2017-06-29` $\rightarrow$ `2017-07-14`)
  - **Fold 3:** `cutoff = 2017-07-14` (đánh giá `2017-07-15` $\rightarrow$ `2017-07-30`)

## 3. Phát hiện chính và Quyết định mô hình hóa

| Thành phần | Phát hiện thực nghiệm | Quyết định |
|---|---|---|
| Cửa sổ huấn luyện (`train_days`) | Cửa sổ `730` ngày (2 năm) và `800` ngày giữ được chu kỳ mùa vụ năm nhưng tránh nhiễu cấu trúc quá cũ | Chọn `train_days = 730` làm chuẩn cho các cấu hình tối ưu |
| Đặc trưng Lịch sử (`history`) | Loại bỏ đặc trưng trễ/cuộn (`lgbm_no_history`) làm điểm RMSLE suy giảm nghiêm trọng | Giữ toàn bộ đặc trưng trễ (`lag_1, 7, 14, 28`) và thống kê cuộn (`mean, std, median`) |
| Đặc trưng Khuyến mãi (`onpromotion`) | Khuyến mãi tương lai biết trước là tín hiệu mạnh nhất cho biến động ngắn hạn | Chuẩn hóa `log1p(onpromotion)` nhân quả theo đúng chỉ mục thời gian |
| Đặc trưng Ngày lễ (`holiday`) | Bổ sung cờ ngày lễ theo phạm vi cửa hàng (`National/Regional/Local`) cải thiện độ ổn định qua các fold | Tích hợp làm bộ đặc trưng cốt lõi (`feature_set = "holiday"`) |
| Đặc trưng Giá dầu (`oil`) | Giá dầu điền từ quá khứ (`<= cutoff`) kết hợp ngày lễ (`holiday_oil`) mang lại mức cải thiện ổn định | Giữ cấu hình `opt_800_holoil_best` làm phương án đối chứng/tổ hợp |
| Độ phức tạp cây & Regularization | Cây quá sâu dễ overfit trên các chuỗi doanh số bằng 0 (`31,62%`); lấy mẫu con giúp giảm phương sai | Chọn `num_leaves = 63`, `min_child_samples = 35`, `subsample = 0.8`, `colsample_bytree = 0.6`, `reg_alpha = 0.1`, `reg_lambda = 1.0` |
| Kiến trúc thuần ML (Non-DL) | LightGBM vượt trội về tốc độ huấn luyện, độ chính xác RMSLE và tính giải thích trên dữ liệu bảng | Loại bỏ toàn bộ khối cấu hình `"lstm"`, chuẩn hóa 100% pipeline thuần ML |

## 4. Tổng hợp kết quả thực nghiệm (3-Fold CV)

| Cấu hình | Vai trò | Đặc trưng | `train_days` | Tham số chính | Ghi nhận đánh giá |
|---|---|---|---|---|---|
| `default` | Baseline LightGBM | `base` | Mặc định | `num_leaves=31`, `lr=0.03` | Mốc tham chiếu cơ sở ban đầu |
| `lgbm_no_history` | Ablation (bỏ lịch sử) | Không lag/roll | 730 | Giống baseline | Điểm số giảm mạnh, khẳng định vai trò tự hồi quy |
| `lgbm_no_promo` | Ablation (bỏ promo) | Không promo | 730 | Giống baseline | Mất khả năng bắt đỉnh doanh số đợt khuyến mãi |
| `holiday_smooth` | Ablation (ngày lễ) | `holiday` | 730 | Làm mượt ngày lễ | Giảm sai số tại các giai đoạn chuyển tiếp lễ |
| `opt_730_samp0806` | Tuning (Subsampling) | `holiday` | 730 | `subsample=0.8`, `colsample=0.6`, `n_est=1200` | Cải thiện rõ rệt độ ổn định giữa 3 folds |
| `opt_800_holoil_best` | Tuning (Holiday + Oil) | `holiday_oil` | 800 | Kết hợp ngày lễ & giá dầu nhân quả | Hiệu quả cao khi mở rộng cửa sổ 800 ngày |
| **`opt_730_l63_mcs35_reg`** | **Cấu hình Quán quân** | **`holiday`** | **730** | **`num_leaves=63`, `mcs=35`, `reg_alpha=0.1`, `reg_lambda=1.0`, `n_est=1400`** | **Đạt cân bằng tốt nhất giữa Mean RMSLE và độ ổn định 3 folds** |

*(Số liệu chi tiết từng lượt chạy, thời gian huấn luyện và điểm từng fold được lưu đầy đủ tại `reports/experiments.csv` và `project_handover/logs/`).*

## 5. Danh sách các file bàn giao

Gói bàn giao được tổ chức thành 2 phần rõ ràng giữa mã nguồn và kết quả thực nghiệm:

src/store_sales/features.py: Trích xuất ma trận đặc trưng dạng bảng (tabular) cho huấn luyện và dự báo.
src/store_sales/models.py: Định nghĩa và huấn luyện mô hình LightGBM thuần Machine Learning.
src/store_sales/run.py: Điều phối huấn luyện, đánh giá 3 folds và xuất log tự động.
configs/*.json: 8 file cấu hình cốt lõi (opt_730_l63_mcs35_r eg, opt_730_samp0806, opt_800_holoil_best, holiday_smooth, default, lgbm_no_history, lgbm_no_promo, smoke)
\reports\lightgbm\Member2_resultsdocs/MEMBER2_BAN_GIAO.md:bảng tổng hợp, dữ liệu cái lần chạy, Checkpoint & Dự báo Validation
 hướng dẫn sử dụng và giới hạn.


## 6. Hướng dẫn tái lập kết quả & Nạp Checkpoint

### Chạy huấn luyện / đánh giá lại bằng CLI
```powershell
# Chạy lại cấu hình quán quân trên một fold cụ thể (ví dụ fold 1: 2017-06-12)
python -m store_sales.run --data "data" --config "configs/opt_730_l63_mcs35_reg.json" --model lightgbm --cutoff 2017-06-12 --out "runs/repro_fold1"

```

### Nạp trực tiếp Checkpoint mô hình đã lưu để dự báo
```python
import numpy as np
import lightgbm as lgb

# Nạp mô hình đã huấn luyện từ thư mục bàn giao
booster = lgb.Booster(
    model_file="reports\lightgbm\Member2_results\Checkpoint & Dự báo Validation\checkpoints/opt_730_l63_mcs35_reg_2017-07-14_lightgbm.txt"
)

# Dự báo trên không gian log1p và chuyển ngược về doanh số gốc
pred_log = np.maximum(0.0, booster.predict(X_test))
pred_sales = np.expm1(pred_log)
```

## 7. Kiểm chứng & Giới hạn

- **Kiểm chứng chống rò rỉ dữ liệu:** Toàn bộ đặc trưng trễ, thống kê cuộn và giá dầu tương lai chỉ sử dụng thông tin tại hoặc trước `cutoff`.
- **Kiểm chứng tái lập:** Mọi cấu hình đều khóa `seed = 42`; chạy lại trên cùng môi trường cho kết quả RMSLE trùng khớp tuyệt đối.
- **Giới hạn hiện tại:** Mô hình sử dụng chiến lược dự báo trực tiếp theo cấu trúc bảng toàn cục (global tabular LightGBM); chưa áp dụng kết hợp (ensembling) đa họ mô hình (CatBoost/XGBoost/Ridge) hoặc hậu xử lý riêng cho nhóm chuỗi có tỷ lệ doanh số bằng 0 kéo dài.
```

Bạn có thể lưu nội dung này thành file `docs/REPORT_SUMMARY.md` (hoặc đặt ngay tại thư mục gốc `REPORT_SUMMARY.md` và trong `project_handover/`) để đồng bộ hoàn toàn với tài liệu `MEMBER1_BAN_GIAO.md` của nhóm.