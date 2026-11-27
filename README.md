# Store Sales Forecasting - INT3405

**VS Code viết code → GitHub lưu code/review → Kaggle chạy, lưu phiên bản và submit.**

Khung dự án cho nhóm 4 người: Seasonal Naive, LightGBM và LSTM dự báo 16 ngày;
kiểm tra dữ liệu/pipeline; lưu run; SHAP hoặc Integrated Gradients cho mô hình tốt nhất.
Workspace hiện có dữ liệu thật và một số kết quả validation cục bộ; dữ liệu thô không được
lưu trong Git và chưa có điểm Kaggle được ghi nhận. Đây chưa phải bài đã hoàn thành.

## Đọc theo thứ tự

1. [Kế hoạch và tiêu chí hoàn thành](docs/KE_HOACH.md).
2. [Thiết lập Windows, VS Code, GitHub và Kaggle](docs/THIET_LAP.md).
3. [Thiết kế thực nghiệm và XAI](docs/THUC_NGHIEM.md).
4. [Hợp đồng dữ liệu và giới hạn starter](docs/DU_LIEU.md).
5. [Báo cáo TV4: Evaluation và XAI](docs/TV4_EVALUATION_XAI.md).

## Bắt đầu trên Windows PowerShell

Giải nén folder, mở folder này bằng VS Code. Cài Python 3.11, Git; chọn đúng Python
interpreter của `.venv`. Nếu máy không có GPU, vẫn viết code và chạy core tests được.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Không cần cài torch/CUDA ở máy cá nhân nếu chỉ train trên Kaggle. Muốn chạy các mô hình
local thì cài thêm `-e ".[ml,dl]"`; PyTorch GPU cần cách cài phù hợp máy.

## Chạy local sau khi có dữ liệu

Tải dữ liệu từ competition vào `data/raw/`. Starter cần `train.csv`, `test.csv`,
`stores.csv`; tải thêm các bảng phụ để EDA và thí nghiệm mở rộng.

```powershell
.\.venv\Scripts\python.exe -m store_sales.run --data data/raw --model baseline --cutoff 2017-07-14 --out outputs/base_fold3
.\.venv\Scripts\python.exe -m store_sales.run --data data/raw --model lightgbm --cutoff 2017-07-14 --out outputs/lgb_fold3
.\.venv\Scripts\python.exe -m store_sales.run --data data/raw --model lstm --cutoff 2017-07-14 --out outputs/lstm_fold3
```

Mỗi lệnh dùng thư mục output MỚI; không ghi đè thí nghiệm cũ. `configs/smoke.json` chỉ
để kiểm tra luồng, không được dùng làm kết quả cuối. Không tự dùng holdout để tuning.

## Đưa source lên Kaggle

```powershell
.\.venv\Scripts\python.exe scripts/build_kaggle_bundle.py
```

Upload `outputs/store-sales-source.zip` làm **Kaggle Dataset private**, chỉ chia sẻ đúng
thành viên trong đội; gắn Dataset này và competition vào notebook. Import
`notebooks/kaggle_runner.ipynb`, làm theo hướng dẫn trong notebook.
Mỗi thay đổi source phải build zip → cập nhật phiên bản Dataset → cập nhật input notebook.
Không sửa code chính thức chỉ trên Kaggle rồi quên đưa về GitHub.

## GitHub

Repo remote CHƯA được tạo trong gói này. Đã có CI, `.gitignore`, PR template và script
tạo repo private; xem [THIET_LAP.md](docs/THIET_LAP.md). Không đưa token hay dữ liệu vào Git.

## Trạng thái kiểm chứng

Xem [KIEM_CHUNG.md](docs/KIEM_CHUNG.md) để biết phần nào đã chạy được trên dữ liệu giả lập,
phần nào còn phải kiểm tra trên Kaggle. Không suy ra điểm hoặc chất lượng mô hình từ unit tests.
