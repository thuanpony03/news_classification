import gc
import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
from typing import Dict, Tuple


class TextClassificationModel:
    def __init__(
            self,
            max_words: int = 50000,  # Increased for large dataset
            max_sequence_length: int = 300,
            embedding_dim: int = 200,
            metadata_embedding_dim: int = 64
    ):
        """
        Initialize the TextClassificationModel.

        Args:
            max_words: Maximum number of words to keep in vocabulary
            max_sequence_length: Maximum length of text sequences
            embedding_dim: Dimension of text embedding
            metadata_embedding_dim: Dimension of metadata embeddings
        """
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.metadata_embedding_dim = metadata_embedding_dim

        # Initialize tokenizer and encoders
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.user_encoder = LabelEncoder()
        self.domain_encoder = LabelEncoder()
        self.label_encoder = LabelEncoder()

        # Initialize vocabulary sizes
        self.n_users = None
        self.n_domains = None
        self.n_labels = None

        # Initialize model and logger
        self.model = None
        self.logger = logging.getLogger(__name__)

        # Training history
        self.history = None

    def fit_encoders(self, train_df: pd.DataFrame):
        """
        Fit all encoders on training data.

        Args:
            train_df: Training DataFrame containing text and metadata
        """
        self.logger.info("Fitting encoders...")

        # Fit text tokenizer
        self.logger.info("Fitting tokenizer on text data...")
        self.tokenizer.fit_on_texts(train_df['full_text'])

        # Fit categorical encoders with unknown categories
        self.logger.info("Fitting label encoder...")
        self.label_encoder.fit(train_df['label'])

        self.logger.info("Fitting user encoder...")
        # Add 'unknown' category to users
        self.user_encoder.fit(pd.concat([pd.Series(['unknown']), train_df['user']]))

        self.logger.info("Fitting domain encoder...")
        # Add 'unknown' category to domains
        self.domain_encoder.fit(pd.concat([pd.Series(['unknown']), train_df['domain']]))

        # Store vocabulary sizes
        self.n_users = len(self.user_encoder.classes_)
        self.n_domains = len(self.domain_encoder.classes_)
        self.n_labels = len(self.label_encoder.classes_)

        # Log vocabulary sizes
        self.logger.info(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        self.logger.info(f"Number of unique users: {self.n_users}")
        self.logger.info(f"Number of unique domains: {self.n_domains}")
        self.logger.info(f"Number of unique labels: {self.n_labels}")

    def prepare_text_data(self, texts: pd.Series) -> np.ndarray:
        """
        Tokenize and pad text data.

        Args:
            texts: Series of text data to process

        Returns:
            Padded sequences of tokenized text
        """
        self.logger.info("Preparing text data...")
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)
        self.logger.info(f"Text data shape: {padded_sequences.shape}")
        return padded_sequences

    def prepare_metadata(
            self,
            users: pd.Series,
            domains: pd.Series,
            days: pd.Series,
            hours: pd.Series,
            is_training: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Prepare metadata features with handling of unknown categories.

        Args:
            users: Series of user data
            domains: Series of domain data
            days: Series of day data
            hours: Series of hour data
            is_training: Whether this is training data

        Returns:
            Dictionary of prepared metadata features
        """
        self.logger.info("Preparing metadata...")

        if is_training:
            # Add an 'unknown' category during training
            self.user_encoder.fit(pd.concat([pd.Series(['unknown']), users]))
            self.domain_encoder.fit(pd.concat([pd.Series(['unknown']), domains]))

            user_encoded = self.user_encoder.transform(users)
            domain_encoded = self.domain_encoder.transform(domains)
        else:
            # Handle unknown categories for validation/test data
            user_encoded = np.array([
                self.user_encoder.transform([user])[0] if user in self.user_encoder.classes_
                else self.user_encoder.transform(['unknown'])[0]
                for user in users
            ])

            domain_encoded = np.array([
                self.domain_encoder.transform([domain])[0] if domain in self.domain_encoder.classes_
                else self.domain_encoder.transform(['unknown'])[0]
                for domain in domains
            ])

        return {
            'user_input': user_encoded,
            'domain_input': domain_encoded,
            'day_input': days.values,
            'hour_input': hours.values
        }

    def create_model(self) -> Model:
        """
        Create the hybrid CNN model architecture.

        Returns:
            Compiled Keras Model
        """
        if self.n_users is None or self.n_domains is None:
            raise ValueError("Must call fit_encoders before creating model")

        # Text input branch
        text_input = layers.Input(shape=(self.max_sequence_length,), name='text_input')
        text_embedding = layers.Embedding(
            self.max_words,
            self.embedding_dim,
            input_length=self.max_sequence_length
        )(text_input)

        # Add spatial dropout to prevent overfitting
        text_embedding = layers.SpatialDropout1D(0.2)(text_embedding)

        # CNN layers with multiple filter sizes
        conv_blocks = []
        filter_sizes = [2, 3, 4, 5]
        n_filters = 256
        for filter_size in filter_sizes:
            conv = layers.Conv1D(
                filters=n_filters,
                kernel_size=filter_size,
                padding='valid',
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(1e-5)
            )(text_embedding)
            pool = layers.GlobalMaxPooling1D()(conv)
            conv_blocks.append(pool)

        text_features = layers.Concatenate()(conv_blocks)
        text_features = layers.Dropout(0.2)(text_features)

        # Metadata inputs
        user_input = layers.Input(shape=(1,), name='user_input')
        domain_input = layers.Input(shape=(1,), name='domain_input')
        day_input = layers.Input(shape=(1,), name='day_input')
        hour_input = layers.Input(shape=(1,), name='hour_input')

        # Metadata embeddings with regularization
        user_embedding = layers.Embedding(
            self.n_users,
            self.metadata_embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )(user_input)
        domain_embedding = layers.Embedding(
            self.n_domains,
            self.metadata_embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )(domain_input)

        # Flatten embeddings
        user_flat = layers.Flatten()(user_embedding)
        domain_flat = layers.Flatten()(domain_embedding)

        # Normalize numerical features
        day_normalized = layers.Lambda(lambda x: x / 7.0)(day_input)
        hour_normalized = layers.Lambda(lambda x: x / 24.0)(hour_input)

        # Combine metadata features
        metadata_features = layers.Concatenate()(
            [user_flat, domain_flat, day_normalized, hour_normalized]
        )

        # Process metadata with deeper network
        metadata_dense1 = layers.Dense(128, activation='relu')(metadata_features)
        metadata_bn1 = layers.BatchNormalization()(metadata_dense1)
        metadata_dropout1 = layers.Dropout(0.3)(metadata_bn1)

        metadata_dense2 = layers.Dense(64, activation='relu')(metadata_dropout1)
        metadata_bn2 = layers.BatchNormalization()(metadata_dense2)
        metadata_features = layers.Dropout(0.2)(metadata_bn2)

        # Combine text and metadata features
        combined_features = layers.Concatenate()([text_features, metadata_features])

        # Dense layers with batch normalization
        dense1 = layers.Dense(512, activation='relu')(combined_features)
        bn1 = layers.BatchNormalization()(dense1)
        dropout1 = layers.Dropout(0.5)(bn1)

        dense2 = layers.Dense(256, activation='relu')(dropout1)
        bn2 = layers.BatchNormalization()(dense2)
        dropout2 = layers.Dropout(0.3)(bn2)

        # Output layer
        output = layers.Dense(self.n_labels, activation='softmax')(dropout2)

        # Create model
        model = Model(
            inputs=[text_input, user_input, domain_input, day_input, hour_input],
            outputs=output
        )

        return model

    def compile_model(self):
        """Compile the model with optimizer and loss function."""
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.logger.info("Model compiled successfully")

    def fit(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            epochs: int = 20,
            batch_size: int = 64
    ):
        """
        Train the model with the provided data.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Training history
        """
        # First fit all encoders on training data
        self.fit_encoders(train_df)

        # Prepare text data
        self.logger.info("Preparing training data...")
        X_train_text = self.prepare_text_data(train_df['full_text'])
        X_val_text = self.prepare_text_data(val_df['full_text'])

        # Prepare metadata
        X_train_meta = self.prepare_metadata(
            train_df['user'],
            train_df['domain'],
            train_df['day_of_week'],
            train_df['hour'],
            is_training=True
        )
        X_val_meta = self.prepare_metadata(
            val_df['user'],
            val_df['domain'],
            val_df['day_of_week'],
            val_df['hour'],
            is_training=False
        )

        # Prepare labels
        self.logger.info("Preparing labels...")
        y_train = tf.keras.utils.to_categorical(
            self.label_encoder.transform(train_df['label'])
        )
        y_val = tf.keras.utils.to_categorical(
            self.label_encoder.transform(val_df['label'])
        )

        # Create and compile model if not exists
        if self.model is None:
            self.logger.info("Creating model...")
            self.model = self.create_model()
            self.compile_model()

        # Create timestamp for model checkpoints
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f'models/model_checkpoint_{timestamp}.weights.h5'

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'logs/fit/{timestamp}',
                histogram_freq=1,
                update_freq='epoch'
            )
        ]

        # Train model
        self.logger.info("Starting model training...")
        self.history = self.model.fit(
            {
                'text_input': X_train_text,
                **X_train_meta
            },
            y_train,
            validation_data=(
                {
                    'text_input': X_val_text,
                    **X_val_meta
                },
                y_val
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Clear memory
        gc.collect()

        return self.history

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on test data.

        Args:
            test_df: Test DataFrame

        Returns:
            Predicted labels
        """
        self.logger.info("Preparing test data for prediction...")

        # Prepare text data
        X_test_text = self.prepare_text_data(test_df['full_text'])

        # Prepare metadata
        X_test_meta = self.prepare_metadata(
            test_df['user'],
            test_df['domain'],
            test_df['day_of_week'],
            test_df['hour'],
            is_training=False
        )

        # Make predictions
        self.logger.info("Making predictions...")
        predictions = self.model.predict(
            {
                'text_input': X_test_text,
                **X_test_meta
            },
            batch_size=64,
            verbose=1
        )

        # Convert predictions to labels
        predicted_labels = self.label_encoder.inverse_transform(predictions.argmax(axis=1))

        return predicted_labels

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            test_df: Test DataFrame

        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info("Evaluating model...")

        # Prepare text data
        X_test_text = self.prepare_text_data(test_df['full_text'])

        # Prepare metadata
        X_test_meta = self.prepare_metadata(
            test_df['user'],
            test_df['domain'],
            test_df['day_of_week'],
            test_df['hour'],
            is_training=False
        )

        # Prepare labels
        y_test = tf.keras.utils.to_categorical(
            self.label_encoder.transform(test_df['label'])
        )

        # Evaluate model
        results = self.model.evaluate(
            {
                'text_input': X_test_text,
                **X_test_meta
            },
            y_test,
            batch_size=64,
            verbose=1
        )

        return dict(zip(self.model.metrics_names, results))

    def save_model(self, path: str):
        """
        Save model to disk.

        Args:
            path: Path to save the model
        """
        self.logger.info(f"Saving model to {path}")
        self.model.save(path)

    def load_model(self, path: str):
        """
        Load model from disk.

        Args:
            path: Path to load the model from
        """
        self.logger.info(f"Loading model from {path}")
        self.model = tf.keras.models.load_model(path)

    def get_training_history(self) -> Dict[str, list]:
        """
        Get training history.

        Returns:
            Dictionary containing training history
        """
        if self.history is None:
            self.logger.warning("No training history available")
            return {}
        return self.history.history


# Usage example:
def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Load data
    logger.info("Loading datasets...")
    data_path = '/home/ponydasierra/thuctap/news_classification/data/processed5'
    train_df = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
    val_df = pd.read_csv(os.path.join(data_path, 'val_data.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test_data.csv'))

    # Preprocess data
    logger.info("Preprocessing data...")
    # Fill NaN values
    train_df['user'] = train_df['user'].fillna('unknown')
    val_df['user'] = val_df['user'].fillna('unknown')
    test_df['user'] = test_df['user'].fillna('unknown')

    train_df['domain'] = train_df['domain'].fillna('unknown')
    val_df['domain'] = val_df['domain'].fillna('unknown')
    test_df['domain'] = test_df['domain'].fillna('unknown')

    # Initialize model
    model = TextClassificationModel(
        max_words=50000,
        max_sequence_length=300,
        embedding_dim=200,
        metadata_embedding_dim=64
    )

    # Train model
    history = model.fit(train_df, val_df, epochs=20, batch_size=64)

    #Train accuracy
    train_accuracy = history.history['accuracy'][-1]
    logger.info(f"Train accuracy: {train_accuracy}")

    #Validation accuracy
    val_accuracy = history.history['val_accuracy'][-1]

    # Evaluate model
    evaluation_results = model.evaluate(test_df)
    logger.info(f"Evaluation results: {evaluation_results}")

    # # Make predictions
    # predictions = model.predict(test_df)
    #
    # # Plot training history
    # model.plot_training_history()

    # Save model
    model.save_model('final_model.keras')

if __name__ == "__main__":
    main()
