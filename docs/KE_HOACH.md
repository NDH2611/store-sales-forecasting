# KẾ HOẠCH CHÍNH THỨC - NHÓM 4 NGƯỜI

## 1. Mục tiêu và phạm vi

Dự báo `sales` theo ngày cho từng `(store_nbr, family)` trong 16 ngày tiếp theo trên
Store Sales - Time Series Forecasting. `sales` là lượng bán, có thể có phần thập phân;
không tự đổi thành doanh thu tiền tệ hoặc làm tròn thành số nguyên.

Đầu ra bắt buộc: code/notebook chạy lại được; đúng file `submission.csv` đã nộp; slide
phân tích dữ liệu, xử lý, phương pháp, thực nghiệm và XAI. Theo đề: 40% leaderboard,
40% báo cáo, 20% trình bày. Không cam kết thứ hạng khi chưa có thực nghiệm.

Hai hướng bắt buộc: **LightGBM (non-DL)** và **LSTM (DL)**. Thêm **Seasonal Naive** làm
baseline. Hai mô hình bắt buộc không đồng nghĩa chỉ được chạy đúng hai thí nghiệm.
Ưu tiên cải thiện kiểm định và đặc trưng trước khi thêm kiến trúc phức tạp.

Ảnh Rules mới nhất của nhóm cho thấy không cần rules acceptance, tối đa 5 thành viên
trên nền tảng và 5 submissions/ngày. Đề môn học yêu cầu nhóm 4 người: giữ đúng 4 người.
Không dựa vào lời hướng dẫn cũ về nút Join/Team. Xác minh trạng thái đội trên tài khoản
thật; nếu giao diện chỉ mở quản lý đội sau submission đầu, chạy baseline hợp lệ trước,
rồi kiểm tra Team/merge. Không bắt cả 4 tài khoản submit nếu giao diện không yêu cầu.

## 2. Ma trận yêu cầu → bằng chứng

| Yêu cầu | Việc phải làm | Bằng chứng để nộp |
|---|---|---|
| Hiểu bài toán | Mục tiêu, đơn vị dự báo, 16 ngày, ý nghĩa RMSLE | Slide bài toán và metric |
| Data processing | EDA, schema, missing, trùng, lịch, joins, biến đổi | Bảng trước/sau; code; lý do giữ/xử lý |
| Non-DL | Baseline + LightGBM; nêu đầu vào và đặc trưng | Config, checkpoint, dự báo validation |
| DL | Global LSTM có cửa sổ lịch sử và đầu ra 16 ngày | Sơ đồ kiến trúc, config, learning curve |
| Thực nghiệm | Chung folds; thống kê kết quả và chi phí | RMSLE từng fold, mean/std; MAE; runtime |
| Vì sao tốt/kém | Ablation, learning curve, lỗi theo nhóm/horizon | Bảng bỏ/thêm đặc trưng; case studies |
| XAI tốt nhất | Chọn theo tiêu chí định trước, không chọn vì dễ giải thích | SHAP/IG; tổng thể + 3 ca; giới hạn |
| Chất lượng dữ liệu | Kiểm tra khóa, phạm vi, thiếu, metadata, lịch | Quality report và tests |
| Chất lượng pipeline | Chống leakage, đúng joins, metric và output | Tests; CI; một người review |
| Tái lập | Source, config, seed, versions, hashes, run ID | Phiên Kaggle chạy sạch + run.json |
| Kết quả Kaggle | Submission từ run được nhận diện rõ | Notebook version, submission, score, thời điểm |
| Đóng góp | Mỗi người có code, thí nghiệm, slide, review | Issues/PR và bảng đóng góp |

Các chương Quality Assurance / Model Quality / Data Quality / Pipeline Quality mà
nhóm cung cấp là khung thực hành, không phải mọi mục đều là yêu cầu triển khai production.
Không cần thêm API, deployment, online monitoring hoặc dashboard để hoàn thành bài này.
Nếu giảng viên đưa rubric riêng sau đó, đối chiếu lại ma trận này.

## 3. Phân công cụ thể

Gán tên thật vào TV1..TV4 trong buổi khởi động.

| Thành viên | Chủ trì | Bàn giao | Review chéo |
|---|---|---|---|
| TV1 - Dữ liệu | EDA toàn bộ CSV; schema; missing; ghép stores/holidays; data contract; tests dữ liệu | 01_eda, data.py, quality report, data dictionary, 3-4 slide | TV4 review keys/joins; TV1 review đầu vào DL |
| TV2 - Non-DL | Baseline; LightGBM; feature snapshots; tuning và ablation; hỗ trợ SHAP nếu thắng | 02_lightgbm, features.py, configs, bảng kết quả, 3-4 slide | TV1 kiểm tra leakage; TV2 review metric/XAI |
| TV3 - DL | LSTM global; input windows; loss; fixed-epoch/tuning; tài nguyên GPU; hỗ trợ IG nếu thắng | 03_lstm, model code, checkpoints, loss/config results, 3-4 slide | TV2 review cửa sổ thời gian; TV3 review final runner |
| TV4 - Evaluation/XAI | Chốt folds/RMSLE; CI; tổng hợp sai số; XAI với chủ mô hình thắng; final runner; ghép slide | 04_xai, evaluate.py, tests pipeline, final notebook, results, 3-4 slide | TV3 review runner; TV1/2 review case studies |

TV4 là điều phối, không viết hộ mọi phần. TV1/2/3 tự viết slide và nhận xét phần mình.
Tất cả dùng cùng data contract và validation; mỗi PR có ít nhất một người review.
TV2/TV3 phải bàn giao cả dự báo theo khóa và checkpoint, không chỉ ảnh điểm số.

## 4. Lộ trình tham chiếu 10 buổi

Chưa có deadline thật: đây là thứ tự công việc, có thể gộp buổi nhưng không bỏ điều kiện
qua mốc. Một buổi là một phiên làm việc nhóm, không cam kết thời gian train.

| Buổi | Công việc | Người chính | Điều kiện hoàn thành |
|---|---|---|---|
| 1 | Chốt đội; GitHub; môi trường; tên thành viên; thời hạn | Cả nhóm | Mở repo, chạy core tests, đọc yêu cầu |
| 2 | EDA, chất lượng dữ liệu, quyết định data processing | TV1 + TV4 | Data contract và quality report; chưa đụng holdout |
| 3 | Pipeline origin-based; baseline; submission kiểm tra luồng | TV2 + TV4 | CSV hợp lệ; Kaggle chấm; lưu phiên notebook |
| 4 | LightGBM đầu tiên và LSTM smoke run | TV2 + TV3 | Cả hai chạy xong một fold nhỏ; không NaN |
| 5 | Chạy đủ 3 tuning folds cho cả hai hướng | TV2 + TV3 | Bảng fold RMSLE, runtime, config, seeds |
| 6 | Tuning nhỏ có giả thuyết; ablation | TV1/2/3 | Có bằng chứng tác dụng đặc trưng/cấu hình |
| 7 | Phân tích lỗi; so sánh công bằng; chọn cấu hình | TV4 + cả nhóm | Chốt model/config/epoch trước holdout |
| 8 | Holdout một lần; XAI mô hình chọn tốt nhất | TV4 + chủ mô hình | Không tune lại; XAI có khóa case và giới hạn |
| 9 | Refit toàn bộ train; clean Kaggle run; nộp cuối | TV4 + TV2/3 | Truy ngược submission về source/config/version |
| 10 | Hoàn thiện slide; tập trình bày; kiểm tra đóng góp | Cả nhóm | Mọi người giải thích được pipeline và kết quả |

## 5. Quy trình kỹ thuật thống nhất

1. Viết/chỉnh sửa trên VS Code. GitHub là nguồn code chính thức.
2. Chạy unit tests, review và merge vào main; ghi SHA commit.
3. Build source zip không chứa dữ liệu/token; cập nhật Kaggle Dataset private của đội.
4. Notebook Kaggle chỉ giải nén source, cài package không thay dependencies ngầm,
   gọi CLI, hiển thị output. Chạy theo đúng config đã lưu.
5. Save Version / Save & Run; giữ output run, checkpoint, environment và config.
6. Ghi summary vào reports/experiments.csv; không chép checkpoint lớn vào GitHub.
7. Nộp từ output đúng phiên; ghi notebook version, score và thời gian trong log.

Không cùng chỉnh một notebook. Ưu tiên code dùng chung trong src. Thay đổi trên Kaggle
chỉ dùng thăm dò; phải đưa về repo và build lại trước khi coi là kết quả chính thức.

## 6. Chiến lược để có báo cáo tốt

- Mỗi nhận xét đi kèm bảng/biểu đồ/thí nghiệm; không khẳng định DL phải thắng.
- Phân biệt giả thuyết và kết luận: 'LSTM overfit' cần loss/validation hoặc thử giảm độ phức tạp.
- Phân tích lỗi theo family, store, ngày dự báo, khuyến mãi và doanh số thấp/cao.
- Báo cáo RMSLE chính; MAE bổ sung theo đơn vị sales. Không dùng accuracy cho regression.
- Mean/std qua folds chỉ là thống kê mô tả; không giả vờ các folds độc lập để tuyên bố ý nghĩa thống kê.
- So sánh lợi ích với chi phí CPU/GPU và thời gian, ghi rõ phần runtime được đo.
- Nếu dữ liệu sạch: ghi kiểm tra đã chạy và số lượng lỗi bằng 0, không bịa bước cleaning.
- Không xóa đỉnh doanh số vì trông 'bất thường' nếu có thể là lễ/khuyến mãi/sự kiện thật.

## 7. Cấu trúc slide đề xuất (14-16 slide)

1. Bài toán và nhóm; 2. Dữ liệu/đơn vị dự báo; 3. EDA;
4. Data processing trước/sau; 5. Chia thời gian và leakage;
6. Baseline; 7. LightGBM/đặc trưng; 8. LSTM/đầu vào/đầu ra;
9. Thiết kế thí nghiệm; 10. Kết quả CV/holdout/Kaggle phân biệt rõ;
11. Ablation; 12. Phân tích lỗi; 13-14. XAI toàn cục và ca cụ thể;
15. Tái lập/chất lượng; 16. Kết luận, hạn chế, đóng góp.

Đưa bảng hyperparameter đầy đủ và thêm case studies vào phụ lục. Thời lượng theo yêu
cầu môn học; hiện chưa có thời lượng chính thức cho bài này.

## 8. Checklist kết thúc

- [ ] Đội 4 người và tên đúng đề, đã kiểm tra trên Kaggle.
- [ ] Có dữ liệu thật; quality report và EDA đã review.
- [ ] Baseline, non-DL, DL đều chạy cùng folds.
- [ ] Có ablation và giải thích hơn/kém bằng bằng chứng.
- [ ] Holdout chỉ dùng sau khi chốt; không dùng leaderboard thay validation.
- [ ] XAI mô hình tốt nhất; output scale, baseline, case keys và giới hạn rõ.
- [ ] Tests CI qua; không có bí mật/dữ liệu trong repo.
- [ ] Kaggle clean run tạo đúng submission cuối; có output/phiên/source/config.
- [ ] Slide đủ các phần; mỗi người có đóng góp và trình bày được.

## Nguồn yêu cầu

- Đề nhóm cung cấp: Final_INT3405E_5.pdf.
- Tài liệu nhóm cung cấp: các chương 14-17 của Machine Learning in Production:
  https://mlip-cmu.github.io/book/14-quality-assurance-basics.html
  https://mlip-cmu.github.io/book/15-model-quality.html
  https://mlip-cmu.github.io/book/16-data-quality.html
  https://mlip-cmu.github.io/book/17-pipeline-quality.html
- Dataset/metric: https://www.kaggle.com/competitions/store-sales-time-series-forecasting
- Trạng thái giao diện Rules: ảnh người dùng cung cấp ngày 14/11/2026; kiểm tra lại
  giao diện trực tiếp trước khi thao tác đội hoặc submit.
