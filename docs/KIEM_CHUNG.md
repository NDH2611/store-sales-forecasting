# Trạng thái kiểm chứng khi bàn giao bộ khung

Ngày tạo: 14/11/2026. Không có dữ liệu competition hoặc kết quả leaderboard trong gói.

## Đã kiểm tra

- 10 unit/integration tests chạy qua bằng unittest trên dữ liệu giả lập:
  RMSLE; duplicate/join; thiếu lịch sử và observed flag; future promotion thiếu;
  bỏ training origin thiếu future promo; cutoff của targets; chống sales leakage;
  baseline trên chu kỳ tuần; căn chỉnh/kiểm tra submission; CLI lưu validation run.
- Toàn bộ source/scripts compile được.
- 5 notebook là JSON hợp lệ, code cells parse được bằng Python AST.
- Source bundle chỉ chứa source/config/metadata, không raw data/token/weights.
- Gói ZIP được đọc lại và kiểm tra CRC trước bàn giao.

## Chưa kiểm tra, cần làm trên Kaggle

- Fit/predict LightGBM và LSTM: môi trường tạo gói chưa có các thư viện này.
- SHAP/Integrated Gradients thực thi trên checkpoint và dữ liệu thật.
- EDA/dải ngày/missing/holiday và các giả định trên dữ liệu gốc.
- GPU/CUDA, runtime và mức RAM thực tế; learning curves validation.
- CI chạy thật trên GitHub; đăng nhập/remote/collaborators chưa được tạo tự động.
- Kaggle Dataset/version input, Save & Run và submission thật.
- Script PowerShell tạo repo: đã soạn, chưa chạy trên Windows/tài khoản người dùng.

## Điều kiện để dùng cho bài nộp

1. Trưởng nhóm tạo GitHub private và mời đúng thành viên.
2. Nhóm chạy baseline trước, xác minh raw files và output.
3. Chạy smoke LightGBM/LSTM; sửa lỗi phụ thuộc nếu có, ghi phiên bản.
4. Hoàn thiện các việc trong KE_HOACH.md, đặc biệt EDA, thí nghiệm và báo cáo.
5. Clean-run notebook Kaggle với source/config khóa, lưu và submit đúng CSV đó.

Tests kiểm tra tính đúng của code ở những tình huống đã xét; không chứng minh mô hình
không có bug, đạt điểm cao hoặc toàn bộ bài đã đáp ứng rubric khi chưa thực nghiệm.
