# Bàn giao Member 1 — Dữ liệu và preprocessing

## 1. Phạm vi công việc

Đã thực hiện:
- Khảo sát các bảng dữ liệu cuộc thi.
- Kiểm tra khóa, giá trị thiếu, giá trị không hợp lệ và độ phủ ngày.
- EDA lượng bán, nhóm hàng, chu kỳ tuần và khuyến mãi.
- Kiểm tra phép ghép metadata cửa hàng.
- Tạo hàm ngày lễ theo ngày–cửa hàng.
- Tạo hàm giá dầu chỉ dùng dữ liệu không muộn hơn cutoff.
- Khảo sát transactions và đối chiếu với lượng bán.
- Kiểm thử và đối chiếu hàm với kết quả EDA thật.

Không bao gồm huấn luyện/tuning mô hình, so sánh mô hình hoặc XAI.

## 2. Phạm vi EDA

- Dữ liệu train gốc: 3.000.888 dòng.
- Giai đoạn EDA: 01/01/2013–14/07/2017.
- Số dòng EDA: 2.943.864.
- Test: 28.512 dòng, 16 ngày từ 16/08–31/08/2017.
- Có 54 cửa hàng, 33 nhóm sản phẩm, 1.782 chuỗi.

Các thống kê sales trong tài liệu này thuộc giai đoạn EDA,
không phải toàn bộ train gốc.

## 3. Phát hiện và quyết định xử lý

| Thành phần | Phát hiện | Quyết định |
|---|---|---|
| Khóa dữ liệu | Không phát hiện trùng ở các khóa đã kiểm tra | Kiểm tra khóa trước xử lý; không cần xóa trùng |
| Sales không hợp lệ | Không phát hiện âm hoặc NaN/inf | Giữ dữ liệu |
| Ngày sales thiếu | Thiếu 25/12/2013, 2014, 2015, 2016 trên toàn bộ chuỗi | Giữ lịch ngày; không tự tạo nhãn sales = 0 |
| Sales bằng 0 | 930.772 dòng, 31,62% trong EDA | Giữ nguyên; phân biệt với thiếu quan sát |
| Phân phối sales | Lệch phải, khác biệt lớn giữa nhóm hàng | log1p là phương án biến đổi; chưa tự cắt giá trị lớn |
| Metadata | Ghép EDA/test đủ, không nhân dòng hay đổi thứ tự id | Ghép many-to-one bằng store_nbr |
| Ngày lễ | Chuyển ngày, phạm vi khác nhau, nhiều sự kiện cùng ngày | Tạo cờ theo phạm vi và gộp trước khi ghép |
| Giá dầu | EDA thiếu 515 ngày, forward fill còn 1 ngày đầu | Điền từ quá khứ, giữ cờ thiếu; không bfill |
| Dầu tương lai | Không mặc định biết tại origin | Phương án ứng viên: giữ giá cuối đã biết |
| Transactions | Thiếu 7.448 ngày–cửa hàng; 118 ngày có sales dương | Không tự điền 0, chưa dùng mặc định |

Không kết luận ngày khai trương từ ngày bán dương đầu tiên.
Không kết luận khuyến mãi gây tăng sales chỉ từ thống kê EDA.

## 4. Các file bàn giao

- notebooks/01_eda.ipynb: EDA, nhận xét và kiểm chứng trên dữ liệu thật.
- src/store_sales/preprocessing.py: hàm xử lý bảng phụ dùng chung.
- tests/test_preprocessing.py: 8 tests mới.
- reports/eda/member1_eda_results/: bảng tổng hợp, biểu đồ và kết quả kiểm chứng.
- docs/MEMBER1_BAN_GIAO.md: hướng dẫn sử dụng và giới hạn.

Kết quả EDA được tạo lại bằng notebook trên Kaggle.
Không cần đưa raw data vào GitHub.

## 5. Cách dùng hàm ngày lễ

```python
from store_sales.preprocessing import build_holiday_features

holiday_features = build_holiday_features(
    holidays,
    stores,
    start="2013-01-01",
    end="2017-08-31",
)
```

Đầu ra duy nhất theo khóa (date, store_nbr), gồm:
- is_holiday
- is_transfer
- is_additional
- is_bridge
- is_work_day
- is_event
- is_transferred_original

Quy tắc:
- National: tất cả cửa hàng.
- Regional: khớp state.
- Local: khớp city.
- Holiday bị chuyển không tính là ngày nghỉ từ riêng sự kiện đó.
- Transfer, Additional, Bridge được tính vào is_holiday.
- Gộp max các cờ trước khi ghép vào sales.

Ví dụ ghép:

```python
result = frame.merge(
    holiday_features,
    on=["date", "store_nbr"],
    how="left",
    validate="many_to_one",
    sort=False,
)
```

Phải kiểm tra lại số dòng, thứ tự id và cờ thiếu sau phép ghép.

is_event chỉ là cờ kiểm tra/ứng viên. Không mặc nhiên dùng sự kiện
đột xuất tương lai làm đầu vào dự báo.

## 6. Cách dùng hàm giá dầu

```python
from store_sales.preprocessing import build_oil_features

oil_features = build_oil_features(
    oil,
    start="2013-01-01",
    end="2017-07-30",
    cutoff="2017-07-14",
)
```

Đầu ra duy nhất theo date:
- oil_price: giá sau forward fill bằng dữ liệu <= cutoff.
- oil_unobserved_at_origin: không có giá quan sát cho ngày đó
  trong thông tin được phép dùng.
- oil_still_missing: chưa có giá quá khứ để điền.
- oil_is_forecast: ngày sau cutoff.

Giả định: giá tại cutoff đã có lúc thực hiện dự báo.
Nếu thời điểm công bố khác, cần lùi mốc dữ liệu được phép dùng.

Phải gọi lại theo từng forecast origin, kể cả các origin trong
quá trình tạo dữ liệu huấn luyện. Không dùng cutoff cuối fold
để tạo đầu vào cho các origin lịch sử sớm hơn.

Không bfill. Không dùng giá dầu thực tế sau cutoff.
Không tự thay NaN ngày đầu thành giá 0.

Nếu LSTM được bổ sung dầu, người phụ trách mô hình cần xử lý
NaN và scaling nhất quán, chỉ fit bằng thông tin train được phép dùng.

## 7. Trạng thái tích hợp

Hai hàm trả về bảng đặc trưng, chưa tự động được đưa vào
load_panel, LightGBM hoặc LSTM.

data.py và features.py hiện giữ nguyên giao diện cũ.

Member 2/3 có thể bổ sung các bảng đặc trưng bằng cấu hình tùy chọn,
kiểm tra kích thước/thứ tự và đánh giá bằng ablation trên cùng folds.

Không ghi báo cáo rằng mô hình đã sử dụng holiday/oil nếu chưa
tích hợp và chạy thực nghiệm.

Transactions chưa có bộ biến đổi đưa vào mô hình; hiện bàn giao
phân tích và chính sách không dùng mặc định.

## 8. Kiểm chứng

- Ruff: All checks passed.
- 18 tests qua: 10 tests cũ và 8 tests preprocessing.
- Dữ liệu thật: 3 phép đối chiếu PASS:
  - Ngày lễ khớp EDA.
  - Dầu lịch sử và cờ thiếu khớp EDA.
  - Dầu dự báo khớp ví dụ EDA.

Các kiểm tra xác nhận tính nhất quán ở những tình huống đã xét,
không chứng minh các đặc trưng sẽ cải thiện điểm dự báo.

## 9. Trách nhiệm tiếp theo

- Member 2: baseline, LightGBM, tích hợp đặc trưng và ablation.
- Member 3: LSTM, chuẩn bị đầu vào mô hình và thực nghiệm.
- Member 4: tổng hợp đánh giá, XAI và báo cáo chung.
- Member 1: giải thích quyết định dữ liệu, sửa lỗi thuộc phần
  dữ liệu nếu phát hiện và hoàn thiện slide phần mình.