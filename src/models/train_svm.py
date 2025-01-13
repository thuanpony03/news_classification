import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import pickle
import logging
from datetime import datetime

# Thiết lập logging cơ bản
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    try:
        # 1. Load dữ liệu
        logger.info("Loading data...")
        train_df = pd.read_csv('/home/ponydasierra/thuctap/news_classification/data/processed5/train_data.csv')
        val_df = pd.read_csv('/home/ponydasierra/thuctap/news_classification/data/processed5/val_data.csv')
        test_df = pd.read_csv('/home/ponydasierra/thuctap/news_classification/data/processed5/test_data.csv')

        # 2. Chuẩn bị dữ liệu
        logger.info("Preparing data...")
        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Sử dụng cả unigrams và bigrams
            min_df=2,  # Bỏ qua các từ xuất hiện ít hơn 2 lần
            max_df=0.95  # Bỏ qua các từ xuất hiện trong >95% documents
        )

        X_train = tfidf.fit_transform(train_df['full_text'])
        X_val = tfidf.transform(val_df['full_text'])
        X_test = tfidf.transform(test_df['full_text'])

        # Label Encoding
        le = LabelEncoder()
        y_train = le.fit_transform(train_df['label'])
        y_val = le.transform(val_df['label'])
        y_test = le.transform(test_df['label'])

        # 3. Tạo và train model
        logger.info("Creating and training SVM model...")
        # Sử dụng LinearSVC vì nó nhanh hơn với dữ liệu lớn
        svm_model = LinearSVC(
            C=1.0,  # Hệ số regularization
            max_iter=1000,  # Số vòng lặp tối đa
            class_weight='balanced'  # Xử lý dữ liệu không cân bằng
        )

        logger.info("Training model...")
        svm_model.fit(X_train, y_train)

        # 4. Đánh giá model
        logger.info("Evaluating model...")
        # Đánh giá trên tập validation
        val_predictions = svm_model.predict(X_val)
        train_predictions = svm_model.predict(X_train)
        val_accuracy = accuracy_score(y_val, val_predictions)
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

        # Đánh giá trên tập test
        test_predictions = svm_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        train_accuracy = accuracy_score(y_train, train_predictions)

        logger.info(f"Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")

        # In báo cáo chi tiết
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, test_predictions))

        # 5. Lưu model và vectorizer
        logger.info("Saving model and vectorizer...")
        model_path = '/home/ponydasierra/thuctap/news_classification/models/'

        # Lưu model
        with open(f"{model_path}svm_model.pkl", 'wb') as f:
            pickle.dump(svm_model, f)

        # Lưu vectorizer
        with open(f"{model_path}svm_tfidf.pkl", 'wb') as f:
            pickle.dump(tfidf, f)

        # Lưu label encoder
        with open(f"{model_path}svm_label_encoder.pkl", 'wb') as f:
            pickle.dump(le, f)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()