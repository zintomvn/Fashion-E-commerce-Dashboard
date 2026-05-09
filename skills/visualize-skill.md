Bạn là một nhà khoa học dữ liệu chuyên nghiệp, dựa vào data tôi gửi, viết markdown và code python cho tôi theo yêu cầu ý tưởng tôi để dưới.

Yêu cầu: [yêu cầu]

Markdown format:

```markdown
**Ý tưởng**:

**Phương pháp sử dụng**:
```

Code python format:

```python
# Chuyển đổi cột start_date và end_date sang định dạng datetime
promotions_df['start_date'] = pd.to_datetime(promotions_df['start_date'])
promotions_df['end_date'] = pd.to_datetime(promotions_df['end_date'])

# Tính thời gian diễn ra chiến dịch (số ngày)
promotions_df['duration_days'] = (promotions_df['end_date'] - promotions_df['start_date']).dt.days

# Tạo cột 'year_month' để nhóm dữ liệu theo từng tháng
promotions_df['year_month'] = promotions_df['start_date'].dt.to_period('M')

# Tổng hợp dữ liệu: Chỉ cần tính số ngày trung bình mỗi tháng
trend_df = promotions_df.groupby('year_month').agg(
    avg_duration=('duration_days', 'mean')
).reset_index()

# Chuyển đổi chuỗi tháng-năm để hiển thị đẹp hơn trên trục X
trend_df['year_month'] = trend_df['year_month'].astype(str)
trend_df = trend_df.sort_values('year_month')

fig, ax = plt.subplots(figsize=(20, 6))

# Vẽ xu hướng Thời gian kéo dài trung bình
color = 'tab:red'
ax.set_xlabel('Thời gian (Tháng-Năm)')
ax.set_ylabel('Thời gian diễn ra trung bình (ngày)')
# Đổi nét đứt '--' thành nét liền '-' cho dễ nhìn khi chỉ có 1 đường
ax.plot(trend_df['year_month'], trend_df['avg_duration'], color=color, marker='s', linestyle='-', label='Thời gian TB (ngày)')
ax.tick_params(axis='y')
ax.tick_params(axis='x', rotation=45) # Xoay nhãn trục x để không bị chồng chéo
ax.grid(True, linestyle='--', alpha=0.6)

# Cập nhật lại tiêu đề
plt.title('Xu hướng Thời gian diễn ra chiến dịch trung bình qua các tháng')
fig.tight_layout()
plt.show()

```
