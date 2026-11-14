# Hợp đồng dữ liệu và quyết định xử lý

## Khóa và đầu ra chung

- Đơn vị: một dòng `(date, store_nbr, family)`; `id` duy nhất.
- `sales` là số thực không âm; không phải class, không làm tròn thành integer.
- Mỗi dự báo có đủ khóa và horizon; submission theo thứ tự `test.csv`, chỉ `id,sales`.
- Mọi người dùng cùng raw files, checksum và mốc validation. Raw files bất biến.

## Starter đã thực hiện

1. Đọc và kiểm tra required columns, null, duplicate keys/id, finite/nonnegative.
2. Dựng daily grid, phát hiện series test chưa thấy; fail nếu cold-start chưa có policy.
3. Ghép stores theo many-to-one, fail nếu trùng metadata hoặc không khớp.
4. Chuyển sales/promotion sang log1p. LSTM dùng log channels, chưa thêm StandardScaler.
5. Missing history trên daily grid: placeholder 0 kèm cờ observed. Đây KHÔNG phải kết
   luận missing sales thật sự bằng 0; phải đếm và phân tích bằng EDA. LightGBM có tỷ
   lệ observed 7/28 ngày, LSTM có observed channel từng ngày.
6. Missing target không được biến thành nhãn 0: ML loại từng dòng nhãn thiếu, LSTM
   loại cửa sổ có target thiếu. Nếu thấy mất dữ liệu nhiều phải sửa policy trước train.
7. Future promotion thiếu tại inference: fail thay vì ngầm điền. Các supervised origins
   lịch sử có future promo thiếu bị bỏ toàn cửa sổ; đếm và ghi tác động bằng EDA trước
   khi chốt dữ liệu train. Promotion trong test là thông tin
   competition cung cấp; validation cũng giả định biết lịch promo tương lai.
8. LightGBM: store/family/metadata được khai báo categorical; lịch sin/cos, horizon,
   promo tương lai, origin-lag 0/6/13/27 và mean-log 7/28 ngày. Mọi sales feature <=origin.
9. LSTM: history `(lookback, 3)` gồm log sales, log promo, observed; store/family
   embeddings; future calendar/promo; đầu ra vector 16. MSE trên log target.
10. Clip dự báo sau expm1 về >=0; nếu NaN/inf thì fail. Không tự che lỗi bằng điền 0.

## TV1 cần hoàn thiện trên dữ liệu thật

| Bảng | Phải kiểm tra | Policy/thí nghiệm |
|---|---|---|
| train/test | Dải ngày thực tế, missing theo ngày/series, 0, phân phối, mùa vụ | Phân biệt 0 thật với missing; báo số lượng trước/sau |
| stores | Keys, city/state/type/cluster | Không biến ID thành đặc trưng số có thứ tự |
| holidays_events | type, locale, locale_name, transferred; nhiều sự kiện/ngày | Map đúng city/state/national, chuyển lễ; aggregate flags trước join |
| oil | Missing cuối tuần, phạm vi ngày, giá trị không hợp lệ | Forward fill từ quá khứ, không bfill/interpolate xuyên validation; thử giữ giá cuối cutoff |
| transactions | Dải ngày và giao dịch thiếu | Chỉ dùng lịch sử tại origin; không join giao dịch thật tương lai |

Starter CHƯA dùng holiday, oil hoặc transactions. Không ghi báo cáo rằng đã xử lý/dùng
các bảng này nếu mới tải về. Không bắt buộc thêm mọi bảng: ghi lý do, thử ablation nếu
khả thi. Dù oil.csv có dữ liệu ở giai đoạn test, phải khai báo rõ giả định khả dụng tại
origin. Để đánh giá thực tế và nhất quán, không mặc nhiên coi giá dầu tương lai đã biết.

Thử StandardScaler cho LSTM là thí nghiệm bổ sung; nếu có, fit trên training history
của từng fold rồi lưu scaler để dùng lại. Không fit chung cả validation/test.

## Bảng preprocessing đưa vào slide

| Vấn đề | Số lượng trước | Quyết định | Số lượng sau | Lý do / ảnh hưởng metric |
|---|---:|---|---:|---|
| Duplicate keys | Chưa đo | Kiểm tra/fail hoặc sửa có căn cứ | Chưa đo | Không bịa số |
| Missing daily grid | Chưa đo | Flag và policy nêu trên | Chưa đo | Phân biệt missing/0 |
| Oil missing | Chưa đo | Nếu dùng: past-only fill | Chưa đo | Ablation |
| Holiday joins | Chưa đo | Locale + aggregate | Chưa đo | Không nhân dòng |
| Outliers | Chưa đo | Kiểm tra sự kiện trước khi sửa | Chưa đo | Không tự xóa peaks |

## Giới hạn của bộ khung

- Chưa EDA hoặc huấn luyện trên dữ liệu competition thật.
- Chưa search hyperparameter, chưa có điểm CV/holdout/leaderboard.
- Chưa có ensemble; chỉ thử sau khi hai mô hình độc lập đúng và có thời gian.
- Chưa có learning curve validation tự động cho LSTM; phải thêm inner split hoặc các
  run fixed-epoch để chẩn đoán, không dùng outer holdout cho early stopping.
- Requirements là khoảng phiên bản để phù hợp Kaggle; run lưu environment chính xác.
  Sau run chính thức, ghi lại bộ phiên bản đã xác nhận và notebook environment.
- CUDA có thể không bitwise deterministic giữa phần cứng/phiên bản; kiểm tra tolerance
  và lưu checkpoint/predictions cuối. Seed không phải bảo đảm tuyệt đối.
