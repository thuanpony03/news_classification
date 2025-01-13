import tensorflow as tf
from tensorflow.keras.layers import (Input, Embedding, Conv1D, MaxPooling1D, LSTM,
                                     Bidirectional, Dense, concatenate, Dropout,
                                     BatchNormalization, Flatten, SpatialDropout1D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import pickle
import os
import json
import re

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(f'hybrid_training_{datetime.now().strftime("%Y%m%d_%H%M")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MetricsHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super(MetricsHistory, self).__init__()
        self.metrics_history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.metrics_history.append(logs)

class TextPreprocessor:
    def __init__(self):
        self.misinformation_patterns = [
            r'<[^>]*>',
            r'\([^)]*không có[^)]*\)',
            r'\[[^]]*không có[^)]*\]'
        ]
        self.word2idx = {}
        self.idx2word = {}
        self.num_words = 0

    def build_vocab(self, texts, min_freq=2):
        """Xây dựng từ điển từ các văn bản đã tokenize"""
        word_freq = {}
        for text in texts:
            if pd.isna(text):
                continue
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1

        vocab = ['<PAD>', '<UNK>']
        vocab.extend([word for word, freq in word_freq.items() if freq >= min_freq])

        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.num_words = len(vocab)

        logger.info(f"Built vocabulary with {self.num_words} words")
        return self.num_words

    def text_to_sequence(self, text):
        """Chuyển văn bản đã tokenize thành chuỗi số"""
        if pd.isna(text):
            return []
        return [self.word2idx.get(word, 1) for word in text.split()]

    def convert_to_sequences(self, texts):
        """Chuyển nhiều văn bản thành chuỗi số"""
        return [' '.join(map(str, self.text_to_sequence(text))) for text in texts]

    def extract_misinformation(self, text):
        """Trích xuất các thông tin không chính xác từ văn bản"""
        if not isinstance(text, str):
            return []

        misinformation = []
        for pattern in self.misinformation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if "không có" in match.group():
                    misinformation.append(match.group())
        return misinformation


class HybridNewsClassifier:
    def __init__(self, max_len=200):
        self.max_len = max_len
        self.model = None

    def prepare_text_sequences(self, texts):
        """Chuẩn bị chuỗi số đã được tokenize"""
        return pad_sequences(
            [list(map(int, text.split())) for text in texts if pd.notna(text)],
            maxlen=self.max_len,
            padding='post',
            truncating='post'
        )

    def create_text_branch(self, input_tensor, vocab_size):
        """Tạo nhánh xử lý văn bản với CNN và LSTM"""
        x = Embedding(
            vocab_size + 1,
            128,
            embeddings_regularizer=l1_l2(l1=1e-5, l2=1e-5)
        )(input_tensor)
        x = SpatialDropout1D(0.2)(x)

        # CNN blocks với kernel sizes khác nhau
        conv_blocks = []
        for kernel_size in [3, 4, 5]:
            conv = Conv1D(
                filters=128,
                kernel_size=kernel_size,
                padding='same',
                activation='relu',
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5)
            )(x)
            conv = BatchNormalization()(conv)
            conv = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv)
            conv_blocks.append(conv)

        x = concatenate(conv_blocks)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = MaxPooling1D(pool_size=1, padding='same')(x)
        return Flatten()(x)

    def create_metadata_branch(self, input_tensor, vocab_size, embedding_dim):
        """Tạo nhánh xử lý metadata"""
        x = Embedding(
            vocab_size + 1,
            embedding_dim,
            embeddings_regularizer=l1_l2(l1=1e-5, l2=1e-5)
        )(input_tensor)

        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=1, padding='same')(x)
        x = Bidirectional(LSTM(32, return_sequences=True))(x)
        return Flatten()(x)

    def create_numerical_branch(self, input_tensor):
        """Tạo nhánh xử lý đặc trưng số"""
        x = Dense(16, activation='relu')(input_tensor)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        return x

    def create_hybrid_model(self, text_vocab_size, metadata_config, num_classes):
        """Tạo mô hình hybrid với các nhánh có thể cấu hình"""
        text_input = Input(shape=(self.max_len,), name='text_input')
        text_features = self.create_text_branch(text_input, text_vocab_size)

        inputs = [text_input]
        features = [text_features]

        for name, config in metadata_config.items():
            if config['type'] == 'categorical':
                input_tensor = Input(shape=(1,), name=f'{name}_input')
                metadata_features = self.create_metadata_branch(
                    input_tensor,
                    config['vocab_size'],
                    config['embedding_dim']
                )
                inputs.append(input_tensor)
                features.append(metadata_features)
            elif config['type'] == 'numerical':
                input_tensor = Input(shape=(config['dim'],), name=f'{name}_input')
                numerical_features = self.create_numerical_branch(input_tensor)
                inputs.append(input_tensor)
                features.append(numerical_features)

        combined = concatenate(features) if len(features) > 1 else features[0]

        x = Dense(256, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        output = Dense(num_classes, activation='softmax')(x)
        return Model(inputs=inputs, outputs=output)


def prepare_metadata(df, type_config):
    """Chuẩn bị metadata dựa trên cấu hình"""
    metadata = {}
    encoders = {}

    if 'user' in type_config:
        user_encoder = LabelEncoder()
        metadata['user'] = user_encoder.fit_transform(df['user'].fillna('unknown'))
        metadata['user_vocab_size'] = len(user_encoder.classes_)
        encoders['user_encoder'] = user_encoder

    if 'domain' in type_config:
        domain_encoder = LabelEncoder()
        metadata['domain'] = domain_encoder.fit_transform(df['domain'].fillna('unknown'))
        metadata['domain_vocab_size'] = len(domain_encoder.classes_)
        encoders['domain_encoder'] = domain_encoder

    if 'time' in type_config:
        metadata['time'] = np.column_stack([
            df['hour'].fillna(0) / 23.0,
            df['day_of_week'].fillna(0) / 6.0
        ])
    if 'title' in type_config:
        metadata['title'] = df['title_seq'].values

    return metadata, encoders


def evaluate_model(model, inputs, y_true, label_encoder, dataset_name=""):
    """Đánh giá hiệu suất mô hình với các metrics chi tiết"""
    y_pred_proba = model.predict(inputs)
    y_pred = np.argmax(y_pred_proba, axis=1)

    loss, accuracy = model.evaluate(inputs, y_true, verbose=0)

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    conf_matrix = confusion_matrix(y_true, y_pred)

    results = {
        'dataset': dataset_name,
        'loss': float(loss),
        'accuracy': float(accuracy),
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist()
    }

    logger.info(f"\nEvaluation Results for {dataset_name} dataset:")
    logger.info(f"Loss: {loss:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    return results




def train_hybrid_model(train_df, val_df, test_df, model_type='text_only'):
    """Train mô hình hybrid với cấu hình chi tiết"""
    logger.info(f"\nStarting training for model type: {model_type}")

    # Khởi tạo preprocessors
    text_processor = TextPreprocessor()
    classifier = HybridNewsClassifier(max_len=200)

    # Xây dựng từ điển từ dữ liệu training
    logger.info("Building vocabulary...")
    _ = text_processor.build_vocab(
        pd.concat([
            train_df['tokenized_content'],
            train_df['tokenized_title']
        ]).dropna()
    )
    logger.info(f"Vocabulary size: {text_processor.num_words}")

    # Chuyển văn bản thành chuỗi số
    logger.info("Converting texts to sequences...")
    train_df['content_seq'] = text_processor.convert_to_sequences(train_df['tokenized_content'])
    val_df['content_seq'] = text_processor.convert_to_sequences(val_df['tokenized_content'])
    test_df['content_seq'] = text_processor.convert_to_sequences(test_df['tokenized_content'])

    # if model_type in ['text_title', 'text_all']:
    #     train_df['title_seq'] = text_processor.convert_to_sequences(train_df['tokenized_title'])
    #     val_df['title_seq'] = text_processor.convert_to_sequences(val_df['tokenized_title'])
    #     test_df['title_seq'] = text_processor.convert_to_sequences(test_df['tokenized_title'])
    # Ensure 'tokenized_title' column is present
    # Ensure 'tokenized_title' column is present
    if model_type in ['text_title', 'text_all']:
        if 'tokenized_title' in train_df.columns:
            train_df['title_seq'] = text_processor.convert_to_sequences(train_df['tokenized_title'])
            val_df['title_seq'] = text_processor.convert_to_sequences(val_df['tokenized_title'])
            test_df['title_seq'] = text_processor.convert_to_sequences(test_df['tokenized_title'])
        else:
            logger.error("tokenized_title column is missing in the DataFrames.")
            raise KeyError("tokenized_title column is missing in the DataFrames.")

    # Phân tích thông tin sai lệch
    logger.info("Analyzing misinformation in the dataset...")
    train_df['misinformation'] = train_df['content'].apply(text_processor.extract_misinformation)
    misinfo_samples = train_df[train_df['misinformation'].apply(len) > 0]
    logger.info(f"\nFound {len(misinfo_samples)} samples containing misinformation")

    # Chuẩn bị đầu vào cho mô hình
    logger.info("Preparing model inputs...")
    X_train_text = classifier.prepare_text_sequences(train_df['content_seq'])
    X_val_text = classifier.prepare_text_sequences(val_df['content_seq'])
    X_test_text = classifier.prepare_text_sequences(test_df['content_seq'])

    # Cấu hình metadata
    metadata_config = {}
    if model_type in ['text_title', 'text_all']:
        metadata_config['title'] = {
            'type': 'categorical',
            'vocab_size': text_processor.num_words,
            'embedding_dim': 64
        }

    if model_type in ['text_user', 'text_all']:
        metadata_config['user'] = {
            'type': 'categorical',
            'vocab_size': len(pd.concat([train_df['user'], val_df['user'], test_df['user']]).unique()),
            'embedding_dim': 32
        }

    if model_type in ['text_domain', 'text_all']:
        metadata_config['domain'] = {
            'type': 'categorical',
            'vocab_size': len(pd.concat([train_df['domain'], val_df['domain'], test_df['domain']]).unique()),
            'embedding_dim': 32
        }

    if model_type in ['text_time', 'text_all']:
        metadata_config['time'] = {
            'type': 'numerical',
            'dim': 2
        }

    # Chuẩn bị metadata
    logger.info("Preparing metadata...")
    train_metadata, encoders = prepare_metadata(train_df, metadata_config)
    val_metadata, _ = prepare_metadata(val_df, metadata_config)
    test_metadata, _ = prepare_metadata(test_df, metadata_config)

    # Chuẩn bị nhãn
    logger.info("Preparing labels...")
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['label'])
    y_val = label_encoder.transform(val_df['label'])
    y_test = label_encoder.transform(test_df['label'])

    # Tạo và biên dịch mô hình
    logger.info("Creating and compiling model...")
    model = classifier.create_hybrid_model(
        text_processor.num_words,
        metadata_config,
        len(label_encoder.classes_)
    )

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Chuẩn bị callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            filepath=f'models/best_hybrid_{model_type}.keras',
            monitor='val_accuracy',
            save_best_only=True
        ),
        MetricsHistory()
    ]

    # Chuẩn bị đầu vào training
    train_inputs = [X_train_text]
    val_inputs = [X_val_text]
    test_inputs = [X_test_text]

    for name in metadata_config:
        train_inputs.append(train_metadata[name])
        val_inputs.append(val_metadata[name])
        test_inputs.append(test_metadata[name])

    # Tính class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    # Training mô hình
    logger.info("Starting model training...")
    history = model.fit(
        train_inputs,
        y_train,
        validation_data=(val_inputs, y_val),
        epochs=20,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Đánh giá mô hình
    logger.info("\nPerforming detailed evaluation...")
    evaluation_results = {
        'train': evaluate_model(model, train_inputs, y_train, label_encoder, "Training"),
        'validation': evaluate_model(model, val_inputs, y_val, label_encoder, "Validation"),
        'test': evaluate_model(model, test_inputs, y_test, label_encoder, "Test")
    }

    # Thêm training history
    evaluation_results['training_history'] = {
        'metrics_history': history.history,
        'final_epoch': len(history.history['loss']),
        'best_val_accuracy': max(history.history['val_accuracy']),
        'best_val_loss': min(history.history['val_loss'])
    }

    return model, evaluation_results, label_encoder, encoders


def main():
    try:
        # Tạo thư mục models nếu chưa tồn tại
        os.makedirs('models', exist_ok=True)

        # Load dữ liệu
        logger.info("Loading datasets...")
        data_path = '/home/ponydasierra/thuctap/news_classification/data/processed5'
        train_df = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
        val_df = pd.read_csv(os.path.join(data_path, 'val_data.csv'))
        test_df = pd.read_csv(os.path.join(data_path, 'test_data.csv'))

        # Log đặc điểm dữ liệu
        logger.info("\nAnalyzing dataset characteristics...")
        logger.info(f"Total training samples: {len(train_df)}")
        logger.info(f"Label distribution:\n{train_df['label'].value_counts()}")

        # Danh sách các loại mô hình cần train
        model_types = [
            # 'text_only',
            'text_title',
            'text_user',
            'text_domain',
            'text_time',
            'text_all'
        ]

        # Lưu kết quả của tất cả các mô hình
        all_results = {}

        for model_type in model_types:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Training {model_type} model...")
            logger.info(f"{'=' * 50}")

            # Train và đánh giá mô hình
            model, evaluation_results, label_encoder, encoders = train_hybrid_model(
                train_df, val_df, test_df, model_type
            )

            # Lưu kết quả
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            model_path = f'models/{model_type}_results_{timestamp}'
            os.makedirs(model_path, exist_ok=True)

            # Lưu mô hình
            model.save(os.path.join(model_path, 'model.keras'))

            # Lưu encoders
            with open(os.path.join(model_path, 'encoders.pkl'), 'wb') as f:
                pickle.dump({'label_encoder': label_encoder, **encoders}, f)

            # Lưu kết quả đánh giá
            with open(os.path.join(model_path, 'evaluation_results.json'), 'w') as f:
                json_results = {
                    k: {
                        'accuracy': v['accuracy'],
                        'loss': v['loss'],
                        'classification_report': v['classification_report'],
                        'confusion_matrix': v['confusion_matrix']
                    }
                    for k, v in evaluation_results.items()
                    if k != 'training_history'
                }
                json_results['training_history'] = evaluation_results['training_history']
                json.dump(json_results, f, indent=4)

            # Lưu kết quả cho so sánh
            all_results[model_type] = {
                'train_accuracy': evaluation_results['train']['accuracy'],
                'val_accuracy': evaluation_results['validation']['accuracy'],
                'test_accuracy': evaluation_results['test']['accuracy'],
                'best_val_accuracy': evaluation_results['training_history']['best_val_accuracy']
            }

            # Log kết quả
            logger.info(f"\nResults for {model_type}:")
            logger.info(f"Train Accuracy: {all_results[model_type]['train_accuracy']:.4f}")
            logger.info(f"Validation Accuracy: {all_results[model_type]['val_accuracy']:.4f}")
            logger.info(f"Test Accuracy: {all_results[model_type]['test_accuracy']:.4f}")
            logger.info(f"Best Validation Accuracy: {all_results[model_type]['best_val_accuracy']:.4f}")

        # So sánh tất cả các mô hình
        logger.info("\n" + "=" * 50)
        logger.info("Final Comparison of All Models:")
        logger.info("=" * 50)

        comparison_df = pd.DataFrame(all_results).transpose()
        logger.info("\n" + str(comparison_df))

        # Lưu kết quả so sánh
        comparison_df.to_csv(f'models/model_comparison_{timestamp}.csv')
        logger.info(f"\nComparison results saved to: models/model_comparison_{timestamp}.csv")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()