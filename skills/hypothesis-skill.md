Bạn là một nhà phân tích dữ liệu và data scientist chuyên nghiệp, dựa vào dữ liệu tôi đưa và hypothesis ở dưới, viết cho tôi code markdown phần giả thuyết (hypothesis) và code kiểm định giả thuyết đó bằng phương pháp phù hợp.

Giả thuyết: [giả thuyết]

Format trình bày markdown:

#### Giả thuyết: [giả thuyết]

**Biến sử dụng:**

**Kiểm định [tên kiểm định]**:

- Giả thuyết H_0: [giả thuyết]
- Giả thuyết H_1: [giả thuyết]

ví dụ:

#### 1. Yếu tố pháp lý (legal_level) ảnh hưởng đến đơn giá (ppm2)

**Biến sử dụng:**

- legal_level: biến phân loại (categorical/ordinal)
- ppm2: biến định lượng liên tục

**Kiểm định ANOVA:** $k$ là số nhóm pháp lý, $\mu_i$ là giá trị trung bình của đơn giá (ppm2) tại nhóm pháp lý thứ $i$, với $i = 1, 2, ..., k$

- Giả thuyết ($H_0$): Không có sự khác biệt có ý nghĩa thống kê về trung bình đơn giá (ppm2) giữa các nhóm pháp lý.

$$
H_0: \mu_1 = \mu_2 = \cdots = \mu_k
$$

- Giả thuyết ($H_1$): Tồn tại ít nhất một cặp nhóm pháp lý có trung bình đơn giá khác nhau. Yếu tố pháp lý (legal_level) có ảnh hưởng đến đơn giá (ppm2).

$$
H_1: \exists \, i \neq j \text{ sao cho } \mu_i \neq \mu_j
$$

Format trình bày code:

```python
hypo_df = df.copy()
df_clean = hypo_df.dropna(subset=['legal_level', 'ppm2'])

```
