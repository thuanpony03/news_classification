import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import logging

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
        # TF-IDF
        tfidf = TfidfVectorizer(max_features=5000)
        X_train = tfidf.fit_transform(train_df['full_text']).toarray()
        X_val = tfidf.transform(val_df['full_text']).toarray()
        X_test = tfidf.transform(test_df['full_text']).toarray()

        # Label Encoding
        le = LabelEncoder()
        y_train = le.fit_transform(train_df['label'])
        y_val = le.transform(val_df['label'])
        y_test = le.transform(test_df['label'])

        # 3. Tạo và train model
        logger.info("Creating and training model...")
        model = Sequential([
            Dense(len(le.classes_), activation='softmax', input_dim=X_train.shape[1])
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=5,
            batch_size=32,
            verbose=1
        )

        # 4. Đánh giá
        logger.info("Evaluating model...")
        loss, accuracy = model.evaluate(X_test, y_test)
        logger.info(f'Test accuracy: {accuracy:.4f}')

        # 5. Lưu model và vectorizer (tùy chọn)
        model.save('/home/ponydasierra/thuctap/news_classification/models/simple_lr.keras')

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()