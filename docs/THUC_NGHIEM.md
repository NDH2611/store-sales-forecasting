# Thiết kế thực nghiệm và XAI

## Chia thời gian

Mốc dưới đây giả định dữ liệu gốc train kết thúc 15/08/2017; xác nhận bằng EDA trước
khi chạy. Tại cutoff t, chỉ sales <=t được coi là đã biết. Dự báo t+1..t+16 cùng lúc.

| Mục đích | Cutoff | Khoảng đánh giá |
|---|---|---|
| Tuning fold 1 | 2017-06-12 | 2017-06-13..2017-06-28 |
| Tuning fold 2 | 2017-06-28 | 2017-06-29..2017-07-14 |
| Tuning fold 3 | 2017-07-14 | 2017-07-15..2017-07-30 |
| Holdout nội bộ khóa | 2017-07-30 | 2017-07-31..2017-08-15 |
| Final refit | 2017-08-15 | Ngày đúng theo test.csv |

Dùng rolling historical training windows trong phần trước mỗi cutoff; starter dùng
365 ngày cho origins, stride 16, lookback 56. Target của mọi training window kết thúc
không muộn hơn cutoff. Đặc trưng lịch sử tính tại origin, không lấy sales thật ở giữa
16 ngày validation. Không random split. Một số window giao nhau về lịch sử là bình
thường; target tương lai tuyệt đối không được rò rỉ.

Starter dùng số epoch/cây cố định, không early-stop bằng outer validation. Muốn early
stopping: chia inner temporal validation nằm hoàn toàn trước outer cutoff, chọn số
epoch/cây, rồi refit trước khi đánh giá outer fold. Không dùng nhãn outer fold để quyết
định thời điểm dừng trong chính lần đánh giá đó.

## Quy tắc lựa chọn

1. Tối ưu mean RMSLE qua 3 tuning folds; kèm từng fold/std và runtime.
2. Nếu chênh lệch rất nhỏ và thiếu ổn định, ưu tiên mô hình đơn giản/nhẹ hơn.
3. Chốt model, feature set, seed, epoch và config; ghi lại trong repo.
4. Chạy holdout một lần để ước lượng cuối. Có thể chạy mọi cấu hình đã khóa trước đó,
   nhưng không tune lại theo holdout; chọn mô hình cuối vẫn dựa quy tắc đã công bố.
5. Nếu phân tích holdout dẫn đến thay đổi phương pháp, phải công bố holdout đã trở
   thành development data; cần giai đoạn mới chưa đụng để có đánh giá độc lập.
6. Refit mô hình chọn trên dữ liệu train được phép dùng, tạo submission.

Leaderboard là đánh giá bổ sung. Không tối ưu đi tối ưu lại dựa trên test score. Không
lấy nhãn test từ nguồn ngoài vì dữ liệu lịch sử đã công khai ở đâu đó.

## Các thí nghiệm cần chạy

| ID | Câu hỏi | So sánh | Người |
|---|---|---|---|
| E0 | Luồng có đúng? | Baseline trên 3 folds | TV2 |
| E1 | ML có vượt baseline? | LightGBM starter full | TV2 |
| E2 | DL có học được? | LSTM starter full | TV3 |
| E3 | Lịch sử hữu ích bao nhiêu? | LightGBM full vs no_history | TV2 |
| E4 | Khuyến mãi hữu ích bao nhiêu? | LightGBM full vs no_promo | TV2 |
| E5 | Bao nhiêu quá khứ là đủ? | LSTM lookback 28 vs 56 | TV3 |
| E6 | LSTM thiếu train hay overfit? | Epoch 5/10/20; thêm validation curve nội bộ | TV3 |
| E7 | Bảng phụ có ích? | Thêm holiday đúng locale/transfer; thử oil riêng | TV1 + TV2 |
| E8 | Kết quả có ổn định? | Config cuối seed 42/123/2026, nếu tài nguyên cho phép | TV2/3 |
| E9 | Đánh giá cuối | Cấu hình đã khóa trên holdout | TV4 |

Trong starter `feature_set=no_history/no_promo` chỉ tác động LightGBM; không dùng tùy
chọn này để tuyên bố đã ablate LSTM. LSTM muốn bỏ promo phải bỏ cả kênh lịch sử và
kênh tương lai nhất quán, đồng thời cập nhật tests/config. Chỉ thay một yếu tố trong
mỗi ablation. Có thể ưu tiên E0..E6 khi gấp, nhưng E7 phải được báo cáo là chưa làm.

## Phân tích kết quả

- Bảng: mô hình, config, RMSLE từng fold, mean/std, MAE, runtime và phần cứng.
- Tách CV, holdout, Kaggle; không gộp thành một cột không rõ nguồn.
- Dùng cùng khóa dòng khi đối chiếu dự báo; không tính metric trên arrays khác thứ tự.
- Biểu đồ dự báo/thực tế theo ngày; lỗi theo horizon 1..16; nhóm hàng/store;
  khuyến mãi và mức sales. Khi tổng hợp doanh số, ghi rõ đó là biểu đồ chẩn đoán,
  không thay thế RMSLE trên từng dòng của competition.
- Xem ca tốt, ca lỗi lớn; nối nguyên nhân có thể với EDA và ablation.
- 'LightGBM xử lý dữ liệu bảng tốt' là giả thuyết chung; kết luận bài phải có E3/E4
  hoặc bằng chứng tương ứng. DL thua vẫn đáp ứng đề nếu làm đúng và giải thích tốt.

## XAI cho mô hình tốt nhất

Xác định 'tốt nhất' theo quy tắc selection ở trên; không chọn cây chỉ vì SHAP tiện.

| Mô hình thắng | Phương pháp | Đầu ra |
|---|---|---|
| LightGBM | Tree SHAP | Global mean abs SHAP, beeswarm, waterfall 3 ca có khóa |
| LSTM | Integrated Gradients có điều kiện | Attribution theo ngày/kênh; global summary; 3 ca; completeness |
| Baseline nếu bất ngờ thắng | Giải thích công thức + đóng góp các ngày lịch sử | Phân rã trung bình same-weekday, so sánh có/bỏ từng quan sát; thêm XAI mô hình học tốt nhất |

Lệnh sau dùng run validation đã lưu:

```text
python -m store_sales.explain --data data/raw --run outputs/lgb_fold3
```

SHAP starter xuất global plot và 3 ví dụ lấy mẫu. Khi làm báo cáo phải bổ sung lựa chọn
3 ca: điển hình, khuyến mãi, sai số lớn; ghi store/family/date/horizon và giá trị thật.
LSTM starter xuất attribution cho 24 series, horizon 1/8/16; TV3/TV4 tạo thêm biểu đồ
từ CSV, so sánh baseline IG thay thế và tăng số bước nếu residual lớn.

Giải thích output thô trên log1p(sales), trước expm1/clipping. IG giữ calendar,
observed flag và identity cố định; chỉ giải thích log sales/promo lịch sử và promo
tương lai, không tuyên bố giải thích embeddings. Mốc 0 là một lựa chọn cần kiểm tra
độ nhạy, không phải chân lý. Các future promo sau ngày đang giải thích có thể ảnh
hưởng dự báo vì mô hình có toàn bộ lịch khuyến mãi 16 ngày tại origin: phải ghi rõ.

Đặc trưng tương quan có thể chia sẻ attribution. Kết quả XAI không chứng minh khuyến
mãi gây tăng sales. Không dừng ở feature importance tích hợp của LightGBM.
