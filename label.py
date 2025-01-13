import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/home/ponydasierra/thuctap/news_classification/data/raw/news_dataset.csv")  # Thay bằng đường dẫn thực tế

# Vẽ biểu đồ phân phối
plt.hist(df['fake_ratio [%]'], bins=50, color='skyblue', edgecolor='black')
plt.title('Phân phối fake_ratio')
plt.xlabel('fake_ratio (%)')
plt.ylabel('Tần suất')
plt.savefig("fake_ratio_distribution.png")

# Thống kê cơ bản
print(df['fake_ratio [%]'].describe())
