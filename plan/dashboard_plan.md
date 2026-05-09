# What‑If Dashboard Plan (DATATHON‑2026 VinUni)

Mục tiêu: xây dashboard tương tác (what‑if) dựa trên **kết quả dự đoán của model** (uplift + counterfactual profit optimization + forecast Revenue/COGS) để kể câu chuyện: _hiện trạng → dự báo → thử kịch bản → ra quyết định_.

---

## 1) Dữ liệu & “AI artifacts” sẽ dùng

### 1.1. Raw tables (data thật)

Nguồn từ `data/raw/`:

- `sales.csv`: chuỗi thời gian **Revenue, COGS** theo ngày (2012–2022)
- `orders.csv`, `order_items.csv`, `payments.csv`: đơn hàng, giỏ hàng, thanh toán
- `products.csv`: category/segment/size/color/price/cogs
- `customers.csv`, `geography.csv`: nhân khẩu học + city/region
- `shipments.csv`: ship*date, delivery_date, shipping_fee → \_lead time*
- `returns.csv`: return_reason, return_quantity, refund_amount
- `inventory.csv`: stock_on_hand, stockout_flag, overstock_flag, stockout_days…
- `web_traffic.csv`: sessions, page_views, traffic_source…
- `promotions.csv`: promo_type, discount_value, stackable_flag, min_order_value…

### 1.2. Output từ model / phân tích (AI integration)

Nguồn từ `outputs/`:

- `uplift_predictions_test_set.csv`: `P_Y_given_Promo`, `P_Y_given_NoPromo`, `Uplift_Score` + nhiều feature (customer/product/channel…)
- `customer_rfm_segments.csv`: gán `customer_segment` (Champions/Loyal/Bargain/At Risk/Hibernating) + RFM
- `campaign_rollout_list.csv`: gợi ý Campaign_Action theo uplift
- `profit_optimization_counterfactual_actions.csv`: baseline vs scenario (profit/revenue/cost) cho các action (price/cogs/stockout…)
- `profit_optimization_best_actions_by_segment.csv`: action tốt nhất theo segment
- `profit_optimization_component_models.csv`: top_importance/top_mi (giải thích yếu tố quan trọng)
- `product_profit_volume_segments.csv`: phân khúc danh mục sản phẩm (Stars/Cash Cows/Dogs/Question Marks)
- Forecast Revenue/COGS có sẵn: `submission_xgb_fold5_tuned.csv` (hoặc `submission_xgb_fold5.csv`)

> Lưu ý: `outputs/models/` hiện chưa có `model_package.joblib`, nên phần forecast trong dashboard sẽ mặc định đọc từ `outputs/submission_xgb_*.csv` (ổn định khi demo). Nếu cần có thể train theo `python -m src.models.train`.

---

## 2) UX dashboard (Streamlit + Plotly) — 3 trang

### Global sidebar filters (dùng chung)

- Filter khách hàng: `customer_segment`, `age_group`, `gender`, `acquisition_channel`, `device_type`, `order_source`, `payment_method`, `city/region`
- Filter sản phẩm: `category`, `segment`, `size`, `color`, price range
- Filter thời gian: date range (cho baseline charts)
- Scenario controls (tuỳ trang): discount/price/cogs %, stockout fix, action selection…

### Page A — **Overview (Baseline & Forecast)**

Mục tiêu: nắm nhanh tình hình kinh doanh + dự báo.
Yêu cầu: dash board có thể điều chỉnh thời gian để xem doanh thu.

- Line: Revenue & COGS theo ngày (train)
- Line: Gross Profit (Revenue − COGS), kèm rolling 7d
- Bar/stack: Order status (delivered/returned) theo tháng
- Box/violin: delivery lead time theo region/city
- Portfolio (scatter/BCG): product profit vs volume + nhãn Stars/Cash Cows…

### Page B — **Promo What‑if (Uplift Targeting)**

Mục tiêu: chọn đúng khách để gửi promo và ước lượng lợi nhuận tăng thêm.

- Inputs:
  - Targeting: chọn segment(s), category/segment, **uplift threshold** hoặc **top‑N**
  - Promo: `promo_type` (percentage/fixed), `discount_value`, `stackable_flag`, `min_order_value` (mô phỏng)
  - Optional: loại nhóm có `return_rate` cao
- Charts:
  - Histogram/violin: phân phối `Uplift_Score` theo segment/category/channel
  - Gain curve: cumulative expected incremental profit theo top‑N / threshold
  - Bar: expected incremental profit theo customer_segment / acquisition_channel
  - Waterfall: baseline → (promo margin) → (uplift) → net
  - Table: danh sách targeting (top customers) + các cột xác suất, uplift, expected profit

### Page C — **Ops What‑if (Price/COGS/Inventory/Shipping/Returns)**

Mục tiêu: thử các hành động vận hành và xem impact profit.

- Inputs:
  - chọn `customer_segment`
  - chọn `action` (tăng/giảm price, giảm cogs, fix stockout_flag, giảm shipping_fee_allocated, giảm return_quantity…)
- Charts:
  - Bar: `delta_profit_pct` theo segment (so sánh nhanh)
  - Waterfall (một segment): baseline_profit → scenario_profit (+ delta_revenue, delta_cost)
  - Leaderboard: best action theo segment
  - Supporting charts (raw) để “giải thích”: stockout_rate trend; return_reason mix; lead time distribution
  - Explainability: top_importance theo segment/target từ component_models

### Page D — **Regression & Confounders (Diagnostics + XAI)**

Mục tiêu: kiểm tra chất lượng dự đoán + tìm **biến gây nhiễu (confounders)** khi đánh giá tác động promo, và giải thích mô hình bằng **Feature Importance + SHAP** (đáp ứng tiêu chí 7 & 8 trong project-final-DV.md).

**Data dùng (ưu tiên “AI artifacts” đã có để demo ổn định):**

- [outputs/uplift_predictions_test_set.csv](outputs/uplift_predictions_test_set.csv): có `True_Y`, `True_T`, `P_Y_given_Promo`, `P_Y_given_NoPromo`, `Uplift_Score` + nhiều feature customer/product/channel…
- (tuỳ chọn) join [outputs/customer_rfm_segments.csv](outputs/customer_rfm_segments.csv) theo `customer_id` để map `customer_segment` giống Page B/C.
- (tuỳ chọn) dùng thêm [outputs/profit_optimization_component_models.csv](outputs/profit_optimization_component_models.csv) để “đối chiếu” top_importance theo segment (nếu muốn nối với Page C).

**Inputs (tận dụng Global filters, thêm tối thiểu):**

- Chọn `target_y` (default: `Uplift_Score`)
  - Option: `Uplift_Score` / `gross_profit_historical` / `True_Y`
- Chọn biến `x_axis` cho scatter (numeric): `price`, `cogs`, `Recency`, `Frequency`, `Monetary`, `promo_order_rate`, `return_rate`…
- Chọn “treatment/outcome pair” để soi confounders:
  - treatment = `True_T`
  - outcome = `True_Y` (hoặc `gross_profit_historical` nếu muốn soi theo profit proxy)

**Charts bắt buộc (đúng spec):**

- **KPI cards: R² / MAE / RMSE**
  - Nếu `target_y = True_Y` (binary): dùng dự đoán theo treatment quan sát:
    - `p_hat_observed = True_T * P_Y_given_Promo + (1-True_T) * P_Y_given_NoPromo`
    - tính R²/MAE/RMSE giữa `True_Y` và `p_hat_observed` (dùng lại hàm trong [src/utils/metrics.py](src/utils/metrics.py)).
  - Nếu `target_y` là continuous (`Uplift_Score`, `gross_profit_historical`): fit nhanh 1 “surrogate regressor” trên subset đã filter (vd LightGBM/Ridge) rồi đánh giá holdout để lấy R²/MAE/RMSE.
- **Correlation**
  - Heatmap correlation trên numeric cols + `{True_T, True_Y, target_y}` để thấy biến nào liên hệ đồng thời với treatment & outcome (gợi ý confounding).
- **Scatter Plot + Regression Line**
  - Scatter `(x_axis, target_y)` + regression line (OLS trendline).
  - (gợi ý để “đúng nghĩa diagnostics”): cho phép đổi sang scatter `(y_true, y_pred)` nếu `target_y=True_Y` (dùng `p_hat_observed`) để nhìn fit.
- **Feature importance**
  - Mode 1: importance của surrogate model (permutation importance hoặc model-based importance).
  - Mode 2 (để “tìm biến gây nhiễu” rõ ràng): **Confounder ranking**
    - với mỗi feature numeric `f`:
      - `assoc_T = |corr(f, True_T)|`
      - `assoc_Y = |corr(f, outcome)|`
      - `confounder_score = assoc_T * assoc_Y`
    - bar chart top‑k theo `confounder_score` (đây là danh sách “biến gây nhiễu” để thảo luận).
- **SHAP plot**
  - SHAP summary plot cho surrogate tree model (TreeExplainer) để giải thích toàn cục (global explainability).
  - Performance guardrail: chỉ compute SHAP trên sample nhỏ (<= 2k rows) + cache.
  - (Optional nâng cao, nếu bạn muốn explain “đúng model uplift”): export/serialize model uplift từ notebook rồi áp dụng đúng ý tưởng T‑learner:
    - `SHAP_Uplift = SHAP_T1 - SHAP_T0` (như phần mô tả XAI trong notebook Promote Analysis).

**Kết nối với các trang khác (để đạt tiêu chí “tích hợp & liên kết”):**

- Page B dùng `Uplift_Score` để target promo → Page D trả lời “Uplift cao vì feature nào?” + “biến nào có thể gây nhiễu khi kết luận promo hiệu quả?”
- Page C dùng counterfactual actions → Page D giúp củng cố niềm tin vào AI bằng metrics + explainability.

---

## 3) What‑if logic (rõ ràng + minh bạch)

### 3.1 Promo what‑if (dựa uplift_predictions_test_set.csv)

- Join `uplift_predictions_test_set.csv` với `customer_rfm_segments.csv` theo `customer_id` để lấy `customer_segment`.
- Mô phỏng lợi nhuận kỳ vọng per (customer, product) theo công thức đơn giản:
  - `base_margin = price - cogs`
  - `promo_price = price*(1-discount%)` (percentage) **hoặc** `price-discount_fixed` (fixed)
  - `promo_margin = promo_price - cogs`
  - `expected_base = P_Y_given_NoPromo * base_margin`
  - `expected_promo = P_Y_given_Promo * promo_margin`
  - `expected_uplift_profit = expected_promo - expected_base`
  - Optional: nhân thêm `(1 - return_rate)` để phạt nhóm return cao
- Sau đó sort theo `expected_uplift_profit` (hoặc uplift) để:
  - lấy Top‑N khách nên gửi
  - vẽ gain curve (cumulative sum)

**Guardrails**

- clamp `promo_price >= 0` (không cho âm)
- hiển thị rõ giả định: **P(Y|Promo/NoPromo) giữ nguyên**, discount chỉ thay đổi margin

### 3.2 Ops what‑if (dựa counterfactual profit optimization)

- Dùng `profit_optimization_counterfactual_actions.csv` làm “bảng tra” baseline/scenario.
- Khi user chọn segment + action → trả về các metric:
  - `baseline_profit`, `scenario_profit`, `delta_profit`, `delta_profit_pct`
  - `delta_revenue`, `delta_cost`

---

## 4) Plan code (triển khai)

### 4.1 Dependencies

- `streamlit`, `plotly`, `pandas`, `numpy`

### 4.2 Cấu trúc code đề xuất (tối giản)

- `streamlit_whatif_dashboard.py` (entrypoint)
- `src/dashboard/data.py`: loaders + caching
- `src/dashboard/metrics.py`: helper tính lead_time/return_rate/gross_profit
- `src/dashboard/scenarios.py`: promo_engine + ops_engine
- `src/dashboard/charts.py`: các hàm vẽ Plotly

_(Có thể dùng 1 file duy nhất nếu muốn demo nhanh; tách module để dễ bảo trì.)_

### 4.3 Steps triển khai

1. Load data + cache (`st.cache_data`) để uplift file lớn vẫn mượt.
2. Build “common filters” ở sidebar.
3. Implement Page A charts (baseline + forecast overlay).
4. Implement Promo page:
   - merge uplift + RFM
   - apply filters + compute expected_uplift_profit
   - gain curve + table targeting
5. Implement Ops page:
   - lookup counterfactual actions
   - delta bars + waterfall
   - explainability chart từ component_models
6. Thêm phần narrative + assumptions trong UI.

---

## 5) Checklist kiểm tra (demo ổn định)

- Sanity: `gross_profit = Revenue - COGS` đúng
- lead_time_days không âm (loại/clip dữ liệu lỗi nếu có)
- promo_price không âm
- Đối chiếu 1–2 dòng `delta_profit` trong Ops page với CSV counterfactual
- Performance: thao tác top‑N/threshold không lag (cache + chỉ compute trên subset đã filter)

---

## 6) Cách kể chuyện khi vấn đáp (gợi ý)

1. Overview: trend + forecast (vì sao cần ra quyết định)
2. Promo: uplift targeting (AI quyết định “gửi cho ai”) + gain curve (trade‑off)
3. Ops: chọn action vận hành (AI counterfactual) + giải thích bằng stockout/returns/shipping + feature importance
