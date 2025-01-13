import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Bidirectional, LSTM, Dense, Dropout, Concatenate
)
from tensorflow.keras.models import Model
import logging
from typing import List, Dict
import numpy as np

class HybridCNNLSTM:
    def __init__(
            self,
            config: Dict,
            vocab_size: int,
            num_classes: int,
            embedding_matrix: np.ndarray = None
    ):
        self.config = config
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_matrix = embedding_matrix
        self.logger = logging.getLogger(__name__)

    def create_embedding_layer(self) -> tf.keras.layers.Layer:
        """Tạo Embedding layer với hoặc không có pre-trained embeddings"""
        if self.embedding_matrix is not None:
            embedding_layer = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.config['embedding_dim'],
                weights=[self.embedding_matrix],
                input_length=self.config['max_sequence_length'],
                trainable=False,
                name='embedding'
            )
            self.logger.info("Using pre-trained embeddings")
        else:
            embedding_layer = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.config['embedding_dim'],
                input_length=self.config['max_sequence_length'],
                name='embedding'
            )
            self.logger.info("Using trainable embeddings")

        return embedding_layer

    def build(self) -> Model:
        """Xây dựng mô hình lai đơn giản"""
        # Input Layer
        input_layer = Input(
            shape=(self.config['max_sequence_length'],),
            name='input'
        )

        # Embedding Layer
        embedding_layer = self.create_embedding_layer()
        embedding_output = embedding_layer(input_layer)

        # Dropout after embedding
        embedding_output = Dropout(
            self.config['embedding_dropout'],
            name='embedding_dropout'
        )(embedding_output)

        # CNN Branch
        conv_outputs = []
        for filter_size in self.config['cnn_filter_sizes']:
            conv = Conv1D(
                filters=self.config['num_filters'],
                kernel_size=filter_size,
                activation='relu',
                padding='valid',
                name=f'conv1d_{filter_size}'
            )(embedding_output)
            pool = GlobalMaxPooling1D(name=f'maxpool_{filter_size}')(conv)
            conv_outputs.append(pool)

        # Merge CNN features
        cnn_features = Concatenate(name='cnn_features')(conv_outputs)

        # LSTM Branch
        lstm_output = Bidirectional(
            LSTM(
                units=self.config['lstm_units'],
                dropout=self.config['dense_dropout'],
            ),
            name='bilstm'
        )(embedding_output)

        # Merge all features
        merged = Concatenate(name='merge_features')([cnn_features, lstm_output])

        # Dense layers
        x = Dense(256, activation='relu', name='dense_1')(merged)
        x = Dropout(self.config['dense_dropout'], name='dropout_1')(x)
        x = Dense(128, activation='relu', name='dense_2')(x)
        x = Dropout(self.config['dense_dropout'], name='dropout_2')(x)

        # Output Layer
        output_layer = Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        )(x)

        # Create model
        model = Model(inputs=input_layer, outputs=output_layer, name='hybrid_cnn_lstm')
        return model


class HybridTrainer:
    def __init__(
            self,
            model: Model,
            config: Dict
    ):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Tạo callbacks cho training"""
        callbacks = []

        # Early Stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True
        )
        callbacks.append(early_stopping)

        # Learning Rate Scheduler
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config['reduce_lr_patience'],
            min_lr=1e-6
        )
        callbacks.append(reduce_lr)

        return callbacks

    def train(
            self,
            train_data: tuple,
            val_data: tuple,
            class_weights: Dict = None
    ) -> Dict:
        """Train model với data đã chuẩn bị"""
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Compile model
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            clipnorm=self.config['clip_norm']
        )

        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train model
        self.logger.info("Starting model training...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            callbacks=self.create_callbacks(),
            class_weight=class_weights,
            verbose=1
        )

        # Save model
        self.model.save(str(self.config['model_path']))

        return history.history