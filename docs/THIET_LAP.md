# Thiết lập Windows → GitHub → Kaggle

## 1. VS Code trên máy cá nhân

1. Giải nén `store-sales-forecasting.zip` vào ví dụ `C:\Projects`.
2. VS Code → File → Open Folder → `C:\Projects\store-sales-forecasting`.
3. Cài Python 3.11, Git; cài extensions được đề xuất trong folder.
4. Mở Terminal PowerShell tại root project:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Gọi thẳng python.exe trong .venv để không phải đổi ExecutionPolicy cho activation.
Ctrl+Shift+P → Python: Select Interpreter → chọn `.venv\Scripts\python.exe`.
Cài local model deps chỉ nếu cần: `python -m pip install -e ".[ml,dl]"` với đúng interpreter.

## 2. Tạo GitHub private

Gói được chuẩn bị sẵn cho GitHub nhưng không chứa repo remote hoặc credential.
Trưởng nhóm tạo một repo private tên `store-sales-forecasting` (không thêm README
khác từ web nếu sẽ push toàn bộ folder này).

Cách nhanh nếu đã có GitHub CLI (`gh`):

```powershell
git config --global user.name "TEN_GITHUB_CUA_BAN"
git config --global user.email "EMAIL_GIT_CUA_BAN"
gh auth login
```

Sau khi đăng nhập, **đọc script trước**, rồi chạy nội dung script theo một trong hai cách:

- Mở `scripts/create_github.ps1`, copy nội dung vào Terminal PowerShell; hoặc
- Nếu chính sách máy cho phép chạy script: `.\scripts\create_github.ps1`.

Script khởi tạo Git, tạo commit đầu, tạo repo PRIVATE trên tài khoản đang đăng nhập
và push. Không chạy lại mù quáng khi bị lỗi: kiểm tra repo/remote trước.
Nếu không dùng gh, tạo empty private repo trên github.com/new rồi:

```powershell
git init -b main
git add .
git commit -m "chore: scaffold reproducible store sales forecasting project"
git remote add origin https://github.com/USERNAME/store-sales-forecasting.git
git push -u origin main
```

Thay USERNAME; nếu đã init/commit/remote thì bỏ bước tương ứng. Cấu hình branch
protection/required CI và một approval nếu gói tài khoản GitHub hỗ trợ; nếu không,
nhóm vẫn thực hiện quy tắc review thủ công.

Trên repo → Settings → Collaborators → thêm đúng 3 tài khoản GitHub còn lại.
Chỉ chia sẻ private source/data với đúng thành viên đội theo rules của cuộc thi.

## 3. Nhánh làm việc

| Người | Nhánh gợi ý |
|---|---|
| TV1 | feature/data-eda |
| TV2 | feature/lightgbm |
| TV3 | feature/lstm |
| TV4 | feature/evaluation-xai |

```powershell
git switch main
git pull
git switch -c feature/data-eda
# Sửa code, chạy tests, chỉ add files cần thiết.
git add src tests docs
git commit -m "feat: validate sales data and document preprocessing"
git push -u origin feature/data-eda
```

Tạo Pull Request trên GitHub, người khác review rồi merge. Sau đó pull main trước
khi làm đầu việc mới. Không commit raw data, .env, kaggle.json, weights, output lớn.
Trước khi merge notebook, xóa output nhạy cảm/lớn và tránh chỉnh cùng file.

## 4. Kaggle chỉ dùng để thực thi

1. Xác minh cả nhóm thuộc đúng đội. Ảnh Rules cho thấy không cần accept rules.
2. Build source từ main đã review:

```powershell
.\.venv\Scripts\python.exe scripts/build_kaggle_bundle.py
```

3. Tạo Kaggle Dataset PRIVATE chứa `outputs/store-sales-source.zip`. Cập nhật dataset
   khi source đổi; chỉ chia sẻ cho đội. Source không có raw data/token.
4. Trong competition, Code → New Notebook hoặc tạo notebook mới rồi gắn competition.
5. Import `notebooks/kaggle_runner.ipynb`; Add Input: competition và source dataset.
6. Baseline/LightGBM có thể chạy CPU; LSTM bật GPU khi tài khoản có tài nguyên.
7. Kiểm tra dependencies trước. Notebook dùng pip `--no-deps` để giữ torch/CUDA hiện có.
   Nếu thiếu LightGBM/SHAP, cài đúng bản và lưu phiên bản; nếu Internet tắt, cần chuẩn bị
   wheels phù hợp hoặc bật Internet nếu tài khoản/cuộc thi cho phép.
8. Chạy smoke trước, sau đó config chính thức. Đặt run ID mới cho mỗi config/fold.
9. Save Version / Save & Run All. Kiểm tra Outputs có run.json, predictions, config,
   environment, checkpoint tương ứng; final mode phải có submission.csv.
10. Nộp file output đúng phiên bằng Submit to Competition khi giao diện cho phép;
    hoặc tải đúng CSV đó và upload ở Submit Prediction. Giữ notebook version và CSV.

Không đưa GitHub token vào notebook để clone private repo. Source Dataset private
giúp chạy lại không phụ thuộc phiên đăng nhập GitHub.

## 5. Khi quản lý đội chưa xuất hiện

Không suy ra trạng thái đội chỉ từ nút Submit Prediction. Ảnh hiện tại không có Team
trong menu và không yêu cầu rules acceptance, nên hướng dẫn cũ 'accept rồi vào Team'
không áp dụng nguyên trạng. Có thể làm baseline và nộp một prediction thật để kiểm
tra luồng rồi xem các mục quản lý participation/team hiện ra; đây là bước dự phòng,
không khẳng định mọi tài khoản đều bắt buộc submit trước.

Nếu vẫn không có Team, chụp giao diện sau submission/participation và hỏi giảng viên
hoặc hỗ trợ Kaggle về cơ chế đội hiện tại. Chưa xác minh đủ 4 người thì không báo cáo
đã lập đội thành công. Việc viết code local không cần chờ bước này.

## 6. Run và output

`outputs/<run_id>/` gồm dự báo có khóa, metrics và slices nếu validation; model nếu ML/DL;
run.json, config.json, quality.json, environment.txt. `total_seconds` là tổng thời gian
đọc/train/predict, KHÔNG phải thời gian train riêng. Nếu cần so chi phí train phải đo
riêng và ghi đúng phần cứng, GPU, số threads.

Ghi summary vào `reports/experiments.csv` (tracked). Giữ output lớn/checkpoint trong
Kaggle version outputs hoặc kho phù hợp của nhóm, ghi đường dẫn tham chiếu trong log.
Không dùng một folder output chung cho nhiều lần chạy.
