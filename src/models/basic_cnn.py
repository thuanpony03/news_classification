import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
import logging
import pickle

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
        # Tokenization
        max_words = 10000  # Số từ tối đa trong từ điển
        max_len = 200  # Độ dài tối đa của mỗi văn bản

        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(train_df['full_text'])

        # Chuyển text thành sequences
        X_train = tokenizer.texts_to_sequences(train_df['full_text'])
        X_val = tokenizer.texts_to_sequences(val_df['full_text'])
        X_test = tokenizer.texts_to_sequences(test_df['full_text'])

        # Pad sequences
        X_train = pad_sequences(X_train, maxlen=max_len)
        X_val = pad_sequences(X_val, maxlen=max_len)
        X_test = pad_sequences(X_test, maxlen=max_len)

        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train = le.fit_transform(train_df['label'])
        y_val = le.transform(val_df['label'])
        y_test = le.transform(test_df['label'])

        # 3. Xây dựng mô hình CNN đơn giản
        logger.info("Building CNN model...")
        model = Sequential([
            # Embedding layer
            Embedding(max_words, 100, input_length=max_len),

            # Convolutional layer
            Conv1D(128, 5, activation='relu'),

            # Global Max Pooling
            GlobalMaxPooling1D(),

            # Output layer
            Dense(len(le.classes_), activation='softmax')
        ])

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 4. Train mô hình
        logger.info("Training model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=5,
            batch_size=32,
            verbose=1
        )

        # 5. Đánh giá mô hình
        logger.info("Evaluating model...")
        # Đánh giá trên tập train
        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        logger.info(f"Train Accuracy: {train_accuracy:.4f}")

        # Đánh giá trên tập validation
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

        # Đánh giá trên tập test
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")

        # 6. Lưu mô hình và tokenizer
        logger.info("Saving model and tokenizer...")
        model_path = '/home/ponydasierra/thuctap/news_classification/models/'

        # Lưu model
        model.save(f"{model_path}simple_cnn.keras")

        # Lưu tokenizer
        with open(f"{model_path}cnn_tokenizer.pkl", 'wb') as f:
            pickle.dump(tokenizer, f)

        # Lưu label encoder
        with open(f"{model_path}cnn_label_encoder.pkl", 'wb') as f:
            pickle.dump(le, f)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()