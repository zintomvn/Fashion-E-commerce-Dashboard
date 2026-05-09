Bạn là một nhà khoa học dữ liệu chuyên nghiệp, dựa vào file notebook [problem_solving.ipynb] đã làm ở trên.

Với mỗi segment đã chia tách ra thành những bài toán nhỏ hơn mỗi mỗi tệp đánh số mục để phân tích

Ví dụ:

# Phân tích theo từng segment

## 1. Phân tích segment 1 - [tên segment]

## 2. Phân tích segment 2 - [tên segment]

Giải bài toán profit bằng cách dùng model biểu diễn quan hệ của các bảng theo 2 target là revenue và cost, giờ hãy hướng dẫn tôi biểu diễn theo revenue = unit_price \* quantity, với mỗi segment trình bày theo format, với mỗi bước phân tích thì đều ghi nhận xét theo format trong file [insight-skill.md]:

## 1. Phân tích segment 1 - [tên segment]

### 1.1. Phân tích Revenue

#### a. Phân tích unit_price

#### b. Phân tích quantity

### 1.2. Phân tích Cost

Cụ thể với phần 1.1. Phân tích Revenue:  
Với phần a. Phân tích unit_price

Chạy model theo các biến và liên kết các bảng sau:

bảng product: price

bảng promotion: promo_id, promo_id_2, promo_type, discount_value, stackable_flag, applicable_category, min_order_value

Với phần b. Phân tích quantity

Chạy model theo các biến và liên kết các bảng sau:

bảng inventory: stock_on_hand, stockout_flag, stockout_days

bảng promotions.csv: min_order_value, discount_value, promo_type, stackable_flag

bảng products.csv: price, category, segment,

Bảng customers.csv: age_group

bảng order_source: order_source

bảng web_traffic: traffic_source

Với phần 1.2. Phân tích Cost

Chạy model theo các biến và liên kết các bảng sau:

Bảng products: cogs

Bảng order_items.csv: quantity

Bảng returns: return_quantity

Bảng orders: order_status, order_date

Bảng inventory: units_received, unit_sold, stock_on_hand, overstock_flag

Bảng shipments: shipping_fee

Với mỗi model đi theo trình tự giống trong file [notebooks/modeling_analysis.ipynb], với mỗi bước thì đều ghi nhận xét theo format trong file [insight-skill.md]:

1. Chạy kiểm định giả thuyết hypothesis theo format trong file hypothesis-skill.md
2. Chạy pipeline modeling
   2.1 Data preprocessing, xem các bảng, data có vấn đề gì về data không, sau đó cleaning, chuẩn hóa, scaling cho phù hợp với từng bảng
   2.2 Feature engineering: chia train/test split, lọc feature importance bằng các kỹ thuật mutual information, RFE, SHAP để lọc không bị bias. Lọc top những features quan trọng ra để lấy feature đó cho modeling bước tiếp theo.
   2.3 Chạy những modeling pipeline là regression. Chạy các mô hình quên thuộc như OLS, hoặc các model regression (list ra và chạy thử từng cái và tối ưu, tuning hyperparameters). Kiểm tra tuyến tính, kiểm tra các thông số metrics.
   2.4 Chạy thêm các model tree-based (XG-Boost, LightGBM, CatBoost, Random Forest) (Cross validation, Tối ưu, tuning cho mỗi model để công bằng) và so sánh với model regression xem cái nào tối ưu hơn. Lấy best model và dự đoán.
   2.5 Giải thích model bằng SHAP và feature importance. Xem những biến nào ảnh hưởng mạnh nhất.
3. Sau khi đã có những feature ảnh hưởng mạnh nhất tới model, chạy những kỹ thuật casual inference, counter factual xem những biến nào có thể thay đổi số phần trăm lợi nhuận (revenue, cost) bao nhiêu (lưu ý đảm bảo với điều kiện thực tế, không thay đổi những phần trăm không có thật), sau đó lọc những thay đổi tối ưu nhất để đưa ra quyết định cho công ty và ghi lại trong markdown theo format giống file [insight-skill.md]

Lưu ý: mỗi segment đều chạy với pipeline như vậy để rút ra insight tối ưu nhất cho từng phân khúc.
