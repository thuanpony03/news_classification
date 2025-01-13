# Part 1: model.py
import os
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
from typing import Dict
import gc
from datetime import datetime


class TextClassificationModel:
    def __init__(
            self,
            feature_combination: str = 'text_user',  # 'text_user', 'text_domain', 'text_time'
            max_words: int = 50000,
            max_sequence_length: int = 300,
            embedding_dim: int = 200,
            metadata_embedding_dim: int = 64
    ):
        self.feature_combination = feature_combination
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.metadata_embedding_dim = metadata_embedding_dim

        # Initialize tokenizer and encoders
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.user_encoder = LabelEncoder()
        self.domain_encoder = LabelEncoder()
        self.label_encoder = LabelEncoder()

        self.n_users = None
        self.n_domains = None
        self.n_labels = None

        self.model = None
        self.logger = logging.getLogger(__name__)

    def fit_encoders(self, train_df: pd.DataFrame):
        """Fit all encoders on training data."""
        self.logger.info("Fitting encoders...")

        # Fit text tokenizer
        self.tokenizer.fit_on_texts(train_df['full_text'])

        # Fit encoders with unknown category
        self.label_encoder.fit(train_df['label'])
        self.user_encoder.fit(pd.concat([pd.Series(['unknown']), train_df['user'].fillna('unknown')]))
        self.domain_encoder.fit(pd.concat([pd.Series(['unknown']), train_df['domain'].fillna('unknown')]))

        # Store sizes
        self.n_users = len(self.user_encoder.classes_)
        self.n_domains = len(self.domain_encoder.classes_)
        self.n_labels = len(self.label_encoder.classes_)

    def prepare_text_data(self, texts: pd.Series) -> np.ndarray:
        """Prepare text data."""
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_sequence_length)

    def prepare_metadata(self, df: pd.DataFrame, is_training: bool = False) -> Dict:
        """Prepare metadata based on feature combination."""
        features = {}

        # Add text features
        features['text_input'] = self.prepare_text_data(df['full_text'])

        if self.feature_combination == 'text_user':
            users = df['user'].fillna('unknown')
            if is_training:
                features['user_input'] = self.user_encoder.transform(users)
            else:
                features['user_input'] = np.array([
                    self.user_encoder.transform([user])[0] if user in self.user_encoder.classes_
                    else self.user_encoder.transform(['unknown'])[0]
                    for user in users
                ])

        elif self.feature_combination == 'text_domain':
            domains = df['domain'].fillna('unknown')
            if is_training:
                features['domain_input'] = self.domain_encoder.transform(domains)
            else:
                features['domain_input'] = np.array([
                    self.domain_encoder.transform([domain])[0] if domain in self.domain_encoder.classes_
                    else self.domain_encoder.transform(['unknown'])[0]
                    for domain in domains
                ])

        elif self.feature_combination == 'text_time':
            features['day_input'] = df['day_of_week'].values
            features['hour_input'] = df['hour'].values

        return features

    def create_model(self) -> Model:
        """Create model architecture based on feature combination."""
        # Text input branch
        text_input = layers.Input(shape=(self.max_sequence_length,), name='text_input')
        text_embedding = layers.Embedding(
            self.max_words,
            self.embedding_dim,
            input_length=self.max_sequence_length
        )(text_input)

        text_embedding = layers.SpatialDropout1D(0.2)(text_embedding)

        # CNN layers
        conv_blocks = []
        for filter_size in [2, 3, 4, 5]:
            conv = layers.Conv1D(
                filters=256,
                kernel_size=filter_size,
                padding='valid',
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(1e-5)
            )(text_embedding)
            pool = layers.GlobalMaxPooling1D()(conv)
            conv_blocks.append(pool)

        text_features = layers.Concatenate()(conv_blocks)
        text_features = layers.Dropout(0.2)(text_features)

        # Metadata branch based on combination
        if self.feature_combination == 'text_user':
            user_input = layers.Input(shape=(1,), name='user_input')
            metadata = layers.Embedding(
                self.n_users,
                self.metadata_embedding_dim,
                embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
            )(user_input)
            metadata = layers.Flatten()(metadata)
            model_inputs = [text_input, user_input]

        elif self.feature_combination == 'text_domain':
            domain_input = layers.Input(shape=(1,), name='domain_input')
            metadata = layers.Embedding(
                self.n_domains,
                self.metadata_embedding_dim,
                embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
            )(domain_input)
            metadata = layers.Flatten()(metadata)
            model_inputs = [text_input, domain_input]

        elif self.feature_combination == 'text_time':
            day_input = layers.Input(shape=(1,), name='day_input')
            hour_input = layers.Input(shape=(1,), name='hour_input')

            day_normalized = layers.Lambda(lambda x: x / 7.0)(day_input)
            hour_normalized = layers.Lambda(lambda x: x / 24.0)(hour_input)

            metadata = layers.Concatenate()([day_normalized, hour_normalized])
            metadata = layers.Dense(32, activation='relu')(metadata)
            model_inputs = [text_input, day_input, hour_input]

        # Combine features
        combined = layers.Concatenate()([text_features, metadata])

        # Dense layers
        x = layers.Dense(256, activation='relu')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # Output
        output = layers.Dense(self.n_labels, activation='softmax')(x)

        return Model(inputs=model_inputs, outputs=output)

    def compile_model(self):
        """Compile the model."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def fit(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            test_df: pd.DataFrame,
            epochs: int = 20,
            batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Train model and return accuracies.

        Returns:
            Dictionary containing train, val, and test accuracies
        """
        # Fit encoders
        self.fit_encoders(train_df)

        # Prepare data
        X_train = self.prepare_metadata(train_df, is_training=True)
        X_val = self.prepare_metadata(val_df, is_training=False)
        X_test = self.prepare_metadata(test_df, is_training=False)

        # Prepare labels
        y_train = tf.keras.utils.to_categorical(self.label_encoder.transform(train_df['label']))
        y_val = tf.keras.utils.to_categorical(self.label_encoder.transform(val_df['label']))
        y_test = tf.keras.utils.to_categorical(self.label_encoder.transform(test_df['label']))

        # Create and compile model
        self.model = self.create_model()
        self.compile_model()

        # Train
        self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.1,
                    patience=3,
                    min_lr=1e-6
                )
            ],
            verbose=1
        )

        # Get accuracies
        train_acc = self.model.evaluate(X_train, y_train, verbose=0)[1]
        val_acc = self.model.evaluate(X_val, y_val, verbose=0)[1]
        test_acc = self.model.evaluate(X_test, y_test, verbose=0)[1]

        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc
        }


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Load data
    logger.info("Loading datasets...")
    data_path = '/home/ponydasierra/thuctap/news_classification/data/processed5'
    train_df = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
    val_df = pd.read_csv(os.path.join(data_path, 'val_data.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test_data.csv'))

    # Train each combination
    feature_combinations = ['text_user', 'text_domain', 'text_time']
    results = {}

    for combination in feature_combinations:
        logger.info(f"\nTraining {combination} model...")

        try:
            model = TextClassificationModel(feature_combination=combination)
            accuracies = model.fit(train_df, val_df, test_df)
            results[combination] = accuracies

            # Clear memory
            del model
            gc.collect()
            tf.keras.backend.clear_session()

        except Exception as e:
            logger.error(f"Error training {combination} model: {str(e)}")
            continue

    # Print results
    logger.info("\nFinal Results:")
    logger.info("=" * 50)
    for combination, accuracies in results.items():
        logger.info(f"\n{combination}:")
        for metric, acc in accuracies.items():
            logger.info(f"{metric}: {acc:.4f}")


if __name__ == "__main__":
    main()