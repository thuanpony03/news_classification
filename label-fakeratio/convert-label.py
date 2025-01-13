import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import os

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'labeling_evaluation_{datetime.now().strftime("%Y%m%d_%H%M")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NewsLabeler:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.kmeans = None

    def load_data(self):
        """Load và kiểm tra dữ liệu"""
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info(f"Loaded dataset with shape: {self.df.shape}")
            logger.info(f"Fake ratio statistics:\n{self.df['fake_ratio [%]'].describe()}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def assign_initial_labels(self, lower_threshold=10, upper_threshold=90):
        """Gán nhãn ban đầu dựa trên ngưỡng"""

        def assign_label(fake_ratio):
            if fake_ratio <= lower_threshold:
                return 'true'
            elif fake_ratio >= upper_threshold:
                return 'false'
            else:
                return 'cluster'

        self.df['temp_label'] = self.df['fake_ratio [%]'].apply(assign_label)

        # Log phân phối nhãn ban đầu
        initial_dist = self.df['temp_label'].value_counts()
        logger.info(f"Initial label distribution:\n{initial_dist}")

    def cluster_middle_data(self, n_clusters=3):
        """Phân cụm dữ liệu ở khoảng giữa"""
        middle_data = self.df[self.df['temp_label'] == 'cluster']['fake_ratio [%]'].values.reshape(-1, 1)

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        middle_labels = self.kmeans.fit_predict(middle_data)

        # Sắp xếp các cụm theo trung tâm
        cluster_centers = self.kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(cluster_centers)
        label_map = {
            sorted_indices[0]: 'ptrue',  # Probably True
            sorted_indices[1]: 'pof',  # Partially True/False
            sorted_indices[2]: 'pfalse'  # Probably False
        }

        self.df.loc[self.df['temp_label'] == 'cluster', 'label'] = [
            label_map[label] for label in middle_labels
        ]

        # Log thông tin về các cụm
        for i, center in enumerate(sorted(cluster_centers)):
            logger.info(f"Cluster {i} center: {center:.2f}")

    def finalize_labels(self):
        """Hoàn thiện nhãn cuối cùng"""
        self.df['label'] = self.df['temp_label'].where(self.df['temp_label'] != 'cluster', self.df['label'])
        self.df.drop(columns=['temp_label'], inplace=True)

    def evaluate_labeling(self):
        """Đánh giá kết quả gán nhãn"""
        # Phân phối nhãn
        label_dist = self.df['label'].value_counts()
        logger.info("\nLabel distribution:")
        logger.info(f"{label_dist}")

        # Thống kê fake_ratio theo nhãn
        fake_ratio_stats = self.df.groupby('label')['fake_ratio [%]'].agg(['mean', 'std', 'min', 'max'])
        logger.info("\nFake ratio statistics by label:")
        logger.info(f"{fake_ratio_stats}")

        # Vẽ box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='label', y='fake_ratio [%]', data=self.df)
        plt.title('Distribution of Fake Ratio by Label')
        plt.savefig('fake_ratio_distribution.png')
        plt.close()

        # Vẽ violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='label', y='fake_ratio [%]', data=self.df)
        plt.title('Violin Plot of Fake Ratio by Label')
        plt.savefig('fake_ratio_violin.png')
        plt.close()

    def save_labeled_data(self):
        """Lưu dữ liệu đã gán nhãn"""
        try:
            self.df.to_csv(self.output_path, index=False)
            logger.info(f"Saved labeled dataset to: {self.output_path}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise


def main():
    # Đường dẫn
    input_path = '/home/ponydasierra/thuctap/news_classification/data/raw/news_dataset.csv'
    output_path = '/home/ponydasierra/thuctap/news_classification/data/raw/dataset_labeled.csv'

    # Tạo thư mục cho output nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        labeler = NewsLabeler(input_path, output_path)

        # Pipeline gán nhãn
        logger.info("Starting labeling process...")
        labeler.load_data()
        labeler.assign_initial_labels()
        labeler.cluster_middle_data()
        labeler.finalize_labels()
        labeler.evaluate_labeling()
        labeler.save_labeled_data()
        logger.info("Labeling process completed successfully!")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()