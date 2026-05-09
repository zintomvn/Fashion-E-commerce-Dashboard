# Ý tưởng và Hướng dẫn Xây dựng Dashboard

## Dashboard: Regression Insights & Explainability

Dashboard này dành cho đội ngũ Data Science và Phân tích Kinh doanh để khám phá **chuyện gì đã xảy ra (Descriptive)** và **tại sao lại xảy ra (Diagnostic)**. Đây cũng là công cụ hoàn hảo để đáp ứng yêu cầu bắt buộc về khả năng giải thích mô hình trong báo cáo kỹ thuật.

- **Model Explainability (Global & Local Interpretability):** Sử dụng biểu đồ SHAP summary plot hoặc Feature Importance để bóc tách mô hình hồi quy. Biểu đồ này sẽ xếp hạng mức độ ảnh hưởng của các biến (như `bounce_rate` từ `web_traffic.csv` hoặc `shipping_fee` từ `shipments.csv`) đến dự báo doanh thu cuối cùng.
- **Customer Segmentation & Price Psychology:** Xây dựng ma trận RFM từ lịch sử giao dịch trong `orders.csv` và `payments.csv`. Trực quan hóa độ nhạy cảm về giá của từng phân khúc khách hàng thông qua mối tương quan giữa `price`, `cogs` và quyết định mua hàng.
- **Anomaly & Revenue Leakage Detection:** Phân tích các điểm nghẽn làm giảm doanh thu thuần. Áp dụng kỹ thuật Negative Sampling để làm nổi bật sự khác biệt giữa các đơn hàng thành công và các giao dịch thất bại (đơn bị `cancelled` hoặc xuất hiện trong `returns.csv`).
