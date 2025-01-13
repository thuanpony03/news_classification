import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Conv1D, GlobalMaxPooling1D,
                                     Dense, LSTM, concatenate, Dropout, BatchNormalization,
                                     Flatten, SpatialDropout1D, GlobalAveragePooling1D)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import logging
from datetime import datetime
import pickle
import os

# Thiết lập logging nâng cao
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    def __init__(self, max_words=15000, max_len=150):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = None

    def prepare_text_data(self, texts, is_training=False):
        """Chuẩn bị dữ liệu văn bản với các tham số tối ưu"""
        if is_training:
            self.tokenizer = Tokenizer(
                num_words=self.max_words,
                oov_token="<OOV>",
                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                lower=True
            )
            self.tokenizer.fit_on_texts(texts)

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be fitted first!")

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding='post',
            truncating='post'
        )

        if is_training:
            logger.info(f"Text vocabulary size: {len(self.tokenizer.word_index)}")
            logger.info(f"Average sequence length: {np.mean([len(seq) for seq in sequences]):.2f}")
            logger.info(f"Max sequence length: {max([len(seq) for seq in sequences])}")

        return padded_sequences


class MetadataPreprocessor:
    def __init__(self):
        self.user_encoder = None
        self.scaler = None

    def prepare_metadata(self, df, is_training=False):
        """Chuẩn bị metadata với xử lý tối ưu"""
        # Normalize time features to [0,1]
        df['hour_norm'] = df['hour'] / 23.0
        df['day_norm'] = df['day_of_week'] / 6.0

        numerical_features = ['hour_norm', 'day_norm']
        numerical_data = df[numerical_features].values

        if is_training:
            self.scaler = StandardScaler()
            self.user_encoder = LabelEncoder()
            numerical_data_scaled = self.scaler.fit_transform(numerical_data)
            user_encoded = self.user_encoder.fit_transform(df['user'].values)
            logger.info(f"Number of unique users: {len(self.user_encoder.classes_)}")
        else:
            if self.scaler is None or self.user_encoder is None:
                raise ValueError("Preprocessors must be fitted first!")

            numerical_data_scaled = self.scaler.transform(numerical_data)
            # Handle unseen users
            user_encoded = df['user'].map(
                lambda x: len(self.user_encoder.classes_) - 1 if x not in self.user_encoder.classes_
                else self.user_encoder.transform([x])[0]
            ).values

        return numerical_data_scaled, user_encoded


def create_improved_model(text_vocab_size, user_vocab_size, num_classes, max_len=150):
    """Tạo mô hình hybrid với kiến trúc tối ưu"""
    # Text branch
    text_input = Input(shape=(max_len,), name='text_input')
    text_embedding = Embedding(
        text_vocab_size,
        64,  # Reduced embedding dimension
        embeddings_regularizer=l1_l2(l1=1e-6, l2=1e-6)
    )(text_input)
    text_embedding = SpatialDropout1D(0.2)(text_embedding)

    # Parallel CNN layers with different kernel sizes
    conv_blocks = []
    for kernel_size in [3, 5]:
        conv = Conv1D(
            filters=64 if kernel_size == 3 else 32,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_regularizer=l1_l2(l1=1e-6, l2=1e-6)
        )(text_embedding)
        conv = BatchNormalization()(conv)

        # Combine max and average pooling
        max_pool = GlobalMaxPooling1D()(conv)
        avg_pool = GlobalAveragePooling1D()(conv)
        conv_blocks.extend([max_pool, avg_pool])

    text_features = concatenate(conv_blocks)
    text_features = Dropout(0.3)(text_features)

    # User branch
    user_input = Input(shape=(1,), name='user_input')
    user_embedding = Embedding(
        input_dim=user_vocab_size,
        output_dim=32,
        embeddings_regularizer=l1_l2(l1=1e-6, l2=1e-6)
    )(user_input)
    user_features = Flatten()(user_embedding)
    user_features = BatchNormalization()(user_features)
    user_features = Dropout(0.2)(user_features)

    # Numerical branch
    num_input = Input(shape=(2,), name='num_input')
    num_dense = Dense(
        16,
        activation='relu',
        kernel_regularizer=l1_l2(l1=1e-6, l2=1e-6)
    )(num_input)
    num_dense = BatchNormalization()(num_dense)
    num_dense = Dropout(0.2)(num_dense)

    # Combine all features
    combined = concatenate([text_features, user_features, num_dense])

    # Dense layers
    dense = Dense(
        64,
        activation='relu',
        kernel_regularizer=l1_l2(l1=1e-6, l2=1e-6)
    )(combined)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.3)(dense)

    # Output layer
    output = Dense(num_classes, activation='softmax')(dense)

    model = Model(inputs=[text_input, user_input, num_input], outputs=output)
    return model




class ModelTrainer:
    def __init__(self, model_path='/home/thuanpony03/thuctap/news_classification/models/'):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)

        self.text_preprocessor = TextPreprocessor(max_words=15000, max_len=150)
        self.metadata_preprocessor = MetadataPreprocessor()
        self.model = None
        self.history = None

    def prepare_callbacks(self):
        """Chuẩn bị callbacks với các tham số tối ưu"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=4,
                restore_best_weights=True,
                min_delta=1e-4
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.model_path, f'best_model_{timestamp}.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

    def compute_class_weights(self, y):
        """Tính toán class weights để xử lý imbalanced data"""
        classes = np.unique(y)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y
        )
        return dict(zip(classes, weights))

    def train(self, train_df, val_df, test_df):
        """Training pipeline với các tối ưu"""
        try:
            logger.info("Starting data preparation...")

            # Text data preparation
            logger.info("Preparing text data...")
            X_train_text = self.text_preprocessor.prepare_text_data(train_df['full_text'], is_training=True)
            X_val_text = self.text_preprocessor.prepare_text_data(val_df['full_text'])
            X_test_text = self.text_preprocessor.prepare_text_data(test_df['full_text'])

            # Metadata preparation
            logger.info("Preparing metadata...")
            X_train_num, X_train_user = self.metadata_preprocessor.prepare_metadata(train_df, is_training=True)
            X_val_num, X_val_user = self.metadata_preprocessor.prepare_metadata(val_df)
            X_test_num, X_test_user = self.metadata_preprocessor.prepare_metadata(test_df)

            # Label preparation
            logger.info("Preparing labels...")
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(train_df['label'])
            y_val = label_encoder.transform(val_df['label'])
            y_test = label_encoder.transform(test_df['label'])

            logger.info(f"Number of classes: {len(label_encoder.classes_)}")
            logger.info(f"Classes: {label_encoder.classes_}")

            # Model creation and compilation
            logger.info("Creating and compiling model...")
            self.model = create_improved_model(
                text_vocab_size=len(self.text_preprocessor.tokenizer.word_index) + 1,
                user_vocab_size=len(self.metadata_preprocessor.user_encoder.classes_),
                num_classes=len(label_encoder.classes_)
            )

            # Compile model
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Calculate class weights
            class_weights = self.compute_class_weights(y_train)
            logger.info(f"Class weights: {class_weights}")

            # Training
            logger.info("Starting training...")
            self.history = self.model.fit(
                [X_train_text, X_train_user, X_train_num],
                y_train,
                validation_data=([X_val_text, X_val_user, X_val_num], y_val),
                epochs=20,
                batch_size=32,  # Reduced batch size
                callbacks=self.prepare_callbacks(),
                class_weight=class_weights,
                verbose=1
            )

            # Model evaluation
            self._evaluate_model(
                [X_train_text, X_train_user, X_train_num],
                y_train,
                "Train"
            )
            self._evaluate_model(
                [X_val_text, X_val_user, X_val_num],
                y_val,
                "Validation"
            )
            self._evaluate_model(
                [X_test_text, X_test_user, X_test_num],
                y_test,
                "Test"
            )

            # Save artifacts
            self._save_artifacts(label_encoder)

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def _evaluate_model(self, X, y, dataset_name):
        """Đánh giá model với logging chi tiết"""
        logger.info(f"\nEvaluating on {dataset_name} set:")
        loss, accuracy = self.model.evaluate(X, y, verbose=0)
        logger.info(f"{dataset_name} Loss: {loss:.4f}")
        logger.info(f"{dataset_name} Accuracy: {accuracy:.4f}")

    def _save_artifacts(self, label_encoder):
        """Lưu model và các preprocessors"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            logger.info("Saving model artifacts...")

            # Save preprocessors
            preprocessors = {
                'tokenizer': self.text_preprocessor.tokenizer,
                'user_encoder': self.metadata_preprocessor.user_encoder,
                'label_encoder': label_encoder,
                'scaler': self.metadata_preprocessor.scaler
            }

            preprocessors_path = os.path.join(self.model_path, f'preprocessors_{timestamp}.pkl')
            with open(preprocessors_path, 'wb') as f:
                pickle.dump(preprocessors, f)
            logger.info(f"Preprocessors saved to: {preprocessors_path}")

            # Save training history
            history_path = os.path.join(self.model_path, f'training_history_{timestamp}.pkl')
            with open(history_path, 'wb') as f:
                pickle.dump(self.history.history, f)
            logger.info(f"Training history saved to: {history_path}")

            logger.info("All artifacts saved successfully!")

        except Exception as e:
            logger.error(f"Error saving artifacts: {str(e)}")
            raise


def main():
    """Main function để chạy training pipeline"""
    try:
        # Set paths
        data_path = '/home/ponydasierra/thuctap/news_classification/data/processed5'
        model_path = '/home/ponydasierra/thuctap/news_classification/models/'

        # Load data
        logger.info("Loading datasets...")
        train_df = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
        val_df = pd.read_csv(os.path.join(data_path, 'val_data.csv'))
        test_df = pd.read_csv(os.path.join(data_path, 'test_data.csv'))

        # Fill NA values
        logger.info("Preprocessing datasets...")
        for df in [train_df, val_df, test_df]:
            df['user'] = df['user'].fillna('unknown')

        # Initialize and train model
        trainer = ModelTrainer(model_path=model_path)
        trainer.train(train_df, val_df, test_df)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()