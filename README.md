# DATATHON 2026 — VinUniversity

> Dự đoán doanh thu (Revenue) và giá vốn hàng bán (COGS) cho nền tảng thương mại điện tử thời trang, kết hợp phân tích lợi nhuận theo phân khúc và tối ưu chiến lược kinh doanh.

---

## Tổng quan dự án

Dự án phân tích toàn diện dữ liệu kinh doanh của một nền tảng thương mại điện tử thời trang, bao gồm:

| Mục tiêu | Mô tả |
|---|---|
| Forecasting | Dự đoán Revenue & COGS theo chuỗi thời gian (time-series) |
| EDA | Phân tích khám phá dữ liệu: khách hàng, sản phẩm, đơn hàng, khuyến mãi, giao vận |
| Profit Analysis | Phân tích lợi nhuận theo phân khúc (segment), tối ưu hóa bằng causal inference |
| Promotion Analysis | Đánh giá hiệu quả chiến dịch khuyến mãi và tác động của web traffic |
| Feature Engineering | Xây dựng đặc trưng lag, rolling window, mã hoá thời gian tuần hoàn |

---

## Cấu trúc thư mục

```
DATATHON-2026-VinUni/
│
├── data/
│   ├── raw/                    # Dữ liệu gốc từ ban tổ chức (14 bảng CSV)
│   ├── processed/              # Dữ liệu đã làm sạch (products_cleaned_final.csv)
│   └── external/               # Dữ liệu bổ sung từ nguồn ngoài
│
├── notebooks/
│   ├── 01_eda.ipynb            # EDA tổng quan toàn bộ dataset
│   ├── 02_feature_engineering.ipynb  # Feature engineering nâng cao
│   ├── 03_modeling.ipynb       # Pipeline modeling baseline
│   ├── 04_promote_analysis.ipynb     # Phân tích chiến dịch khuyến mãi
│   ├── 05_profit_analysis.ipynb      # Phân tích lợi nhuận theo segment
│   ├── modeling_baseline.ipynb       # Baseline model XGBoost/LightGBM
│   └── basic eda/              # EDA chi tiết từng bảng (13 notebooks)
│       ├── 1_products.ipynb
│       ├── 2_customers.ipynb
│       ├── 3_promotions.ipynb
│       ├── 4_geography.ipynb
│       ├── 5_orders.ipynb
│       ├── 6_order_items.ipynb
│       ├── 7_payments.ipynb
│       ├── 8_shipments.ipynb
│       ├── 9_returns.ipynb
│       ├── 10_reviews.ipynb
│       ├── 11_sales.ipynb      # EDA chuỗi thời gian doanh thu
│       ├── 12_inventory.ipynb
│       └── 13_web_traffic.ipynb
│
├── src/                        # Module Python tái sử dụng
│   ├── data/
│   │   ├── load_data.py        # Tải tất cả bảng CSV theo tên logic
│   │   └── preprocess.py       # Làm sạch và đảm bảo liên tục chuỗi thời gian
│   ├── features/
│   │   └── build_features.py   # Time features, lag features, rolling features
│   ├── models/
│   │   ├── train.py            # CLI training: HistGradientBoosting cho Revenue & COGS
│   │   └── predict.py          # CLI inference: dự đoán autoregressive
│   └── utils/
│       └── metrics.py          # MAE, RMSE, R2 metrics
│
├── outputs/
│   ├── models/                 # Model đã huấn luyện (.joblib)
│   ├── submissions/            # File nộp bài (CSV)
│   ├── figures/                # Biểu đồ xuất ra
│   ├── *.html                  # Notebook đã render thành HTML
│   └── *.csv                   # Kết quả phân tích: RFM, association rules, profit
│
├── skills/                     # Hướng dẫn prompt và format chuẩn
│   ├── pipeline-skill.md       # Quy trình phân tích segment profit
│   ├── insight-skill.md        # Format viết nhận xét / insight
│   ├── visualize-skill.md      # Format viết code visualization
│   ├── hypothesis-skill.md     # Format kiểm định giả thuyết
│   └── question-skill.md       # Framework đặt câu hỏi nghiên cứu
│
├── requirements.txt            # Thư viện Python cần thiết
└── README.md
```

---

## Dữ liệu (`data/raw/`)

| Bảng | Kích thước | Mô tả |
|---|---|---|
| `orders.csv` | ~44 MB | Thông tin đơn hàng (order_id, order_date, order_source, order_status) |
| `order_items.csv` | ~23 MB | Chi tiết sản phẩm trong đơn (quantity, unit_price, product_id) |
| `customers.csv` | ~7 MB | Thông tin khách hàng (age_group, signup_date) |
| `shipments.csv` | ~19 MB | Vận chuyển (ship_date, delivery_date, shipping_fee) |
| `payments.csv` | ~18 MB | Thanh toán (payment_method, payment_status) |
| `returns.csv` | ~2 MB | Trả hàng (return_date, return_quantity) |
| `inventory.csv` | ~5 MB | Tồn kho (stock_on_hand, stockout_flag, overstock_flag) |
| `reviews.csv` | ~6 MB | Đánh giá sản phẩm (rating, review_date) |
| `products.csv` | ~190 KB | Danh mục sản phẩm (price, cogs, category, segment) |
| `promotions.csv` | ~4 KB | Chiến dịch khuyến mãi (promo_type, discount_value, stackable_flag) |
| `geography.csv` | ~1.4 MB | Địa lý (region, city, province) |
| `sales.csv` | ~126 KB | **[TARGET]** Doanh thu & COGS theo ngày (Date, Revenue, COGS) |
| `web_traffic.csv` | ~204 KB | Lưu lượng truy cập web (date, traffic_source, sessions) |
| `sample_submission.csv` | ~18 KB | Template nộp bài (Date) |

---

## Cài đặt môi trường

### 1. Yêu cầu hệ thống

- Python 3.9+
- pip hoặc conda
- Jupyter Notebook / JupyterLab

### 2. Tạo môi trường ảo

```powershell
# Tạo virtual environment
python -m venv .venv

# Kích hoạt (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Kích hoạt (Windows CMD)
.\.venv\Scripts\activate.bat
```

### 3. Cài đặt thư viện

```powershell
pip install -r requirements.txt
```

Các thư viện chính: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `shap`, `joblib`, `matplotlib`, `seaborn`, `statsmodels`, `scipy`

---

## Cách chạy

### A. Chạy Notebook phân tích

```powershell
# Mở JupyterLab từ thư mục gốc dự án
jupyter lab

# Hoặc Jupyter Notebook cổ điển
jupyter notebook
```

Thứ tự chạy notebook khuyến nghị:

```
notebooks/basic eda/1_products.ipynb       -> EDA san pham
notebooks/basic eda/2_customers.ipynb      -> EDA khach hang
notebooks/basic eda/11_sales.ipynb         -> EDA chuoi thoi gian doanh thu
notebooks/01_eda.ipynb                     -> EDA tong hop
notebooks/02_feature_engineering.ipynb    -> Feature engineering
notebooks/03_modeling.ipynb               -> Modeling baseline
notebooks/04_promote_analysis.ipynb       -> Phan tich khuyen mai
notebooks/05_profit_analysis.ipynb        -> Phan tich loi nhuan theo segment
```

---

### B. Train Model (CLI)

Train mô hình dự báo Revenue & COGS bằng `HistGradientBoostingRegressor`:

```powershell
# Chay voi tham so mac dinh
python -m src.models.train

# Chay voi tham so tuy chinh
python -m src.models.train `
    --raw-dir data/raw `
    --model-out outputs/models/model_package.joblib `
    --val-days 365 `
    --lags 1,7,14,28,56,365 `
    --windows 7,14,28 `
    --random-state 42
```

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--raw-dir` | `data/raw` | Thư mục chứa CSV gốc |
| `--model-out` | `outputs/models/model_package.joblib` | Nơi lưu model |
| `--val-days` | `365` | Số ngày dùng để validation |
| `--lags` | `1,7,14,28,56,365` | Các lag features (ngày) |
| `--windows` | `7,14,28` | Rolling window sizes |
| `--random-state` | `42` | Random seed |

Output khi train thành công:

```
[Revenue] val_mae=... val_rmse=... val_r2=...
[COGS]    val_mae=... val_rmse=... val_r2=...
Saved model package -> outputs/models/model_package.joblib
```

---

### C. Tạo file Submission (CLI)

> Phải chạy **train** trước khi predict.

```powershell
# Chay voi tham so mac dinh
python -m src.models.predict

# Chay voi tham so tuy chinh
python -m src.models.predict `
    --raw-dir data/raw `
    --template data/raw/sample_submission.csv `
    --model-path outputs/models/model_package.joblib `
    --out-path outputs/submissions/submission.csv
```

Output:

```
Wrote submission -> outputs/submissions/submission.csv
```

---

## Pipeline phân tích lợi nhuận (Profit Analysis)

Theo chuẩn `skills/pipeline-skill.md`, mỗi phân khúc (segment) chạy pipeline đầy đủ:

```
1. Kiem dinh gia thuyet (Hypothesis Testing)
   Theo format: skills/hypothesis-skill.md

2. Modeling Pipeline
   2.1 Data Preprocessing & Cleaning
   2.2 Feature Engineering (Mutual Information, RFE, SHAP)
   2.3 Regression Models (OLS, Ridge, Lasso, ...)
   2.4 Tree-based Models (XGBoost, LightGBM, CatBoost, Random Forest)
       Cross-validation + Hyperparameter tuning
   2.5 SHAP Explanation & Feature Importance

3. Causal Inference & Counterfactual Analysis
   De xuat hanh dong toi uu loi nhuan
```

Phân tích Revenue = `unit_price x quantity`:

| Target | Bảng nguồn | Biến quan trọng |
|---|---|---|
| `unit_price` | `products`, `promotions` | price, promo_type, discount_value, stackable_flag |
| `quantity` | `inventory`, `promotions`, `products`, `customers`, `web_traffic` | stock_on_hand, stockout_flag, age_group, traffic_source |
| `cost` | `products`, `order_items`, `returns`, `inventory`, `shipments` | cogs, return_quantity, shipping_fee, overstock_flag |

---

## Kết quả chính (`outputs/`)

| File | Mô tả |
|---|---|
| `customer_rfm_segments.csv` | Phân khúc RFM của toàn bộ khách hàng |
| `product_profit_volume_segments.csv` | Phân khúc sản phẩm theo lợi nhuận và khối lượng |
| `profit_optimization_best_actions_by_segment.csv` | Hành động tối ưu lợi nhuận tốt nhất theo phân khúc |
| `profit_optimization_counterfactual_actions.csv` | Phân tích counterfactual: thay đổi bao nhiêu % lợi nhuận |
| `campaign_rollout_list.csv` | Danh sách chiến dịch khuyến mãi đề xuất triển khai |
| `association_rules_*.csv` | Association rules: khách hàng theo danh mục/sản phẩm |
| `submission_xgb_fold5_tuned.csv` | File nộp bài cuối cùng (XGBoost tuned) |

---

## Skills / Prompt Templates (`skills/`)

| File | Mục đích |
|---|---|
| `pipeline-skill.md` | Quy trình phân tích segment profit chuẩn |
| `insight-skill.md` | Format viết nhận xét dữ liệu (Quan sát - Insights - Gợi ý hành động) |
| `visualize-skill.md` | Format viết code Python visualization |
| `hypothesis-skill.md` | Framework kiểm định giả thuyết thống kê |
| `question-skill.md` | Framework đặt câu hỏi nghiên cứu |

---

## Module `src/` — API nội bộ

```python
from src.data.load_data import load_table, load_sales, load_all_raw

# Load mot bang theo ten logic
orders = load_table("orders")

# Load bang sales (target variable)
sales = load_sales()  # -> DataFrame[Date, Revenue, COGS]

# Load toan bo 14 bang
all_tables = load_all_raw()
```

```python
from src.features.build_features import build_supervised_frame

# Tao frame supervised learning voi time + lag + rolling features
frame = build_supervised_frame(
    sales,
    target_col="Revenue",
    lags=[1, 7, 14, 28, 56, 365],
    windows=[7, 14, 28]
)
```

---

## Nhóm

VinUniversity Datathon 2026 — Phân tích dữ liệu thương mại điện tử thời trang

---

## License

Dự án phục vụ mục đích học thuật và thi đấu nội bộ. Không phát hành thương mại.
