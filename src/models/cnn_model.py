import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout, Concatenate, BatchNormalization,
    LayerNormalization
)
from tensorflow.keras.models import Model
import logging
from typing import List, Dict
import numpy as np


class TextCNN:
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
                trainable=False
            )
            self.logger.info("Created embedding layer with pre-trained weights")
        else:
            embedding_layer = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.config['embedding_dim'],
                input_length=self.config['max_sequence_length']
            )
            self.logger.info("Created trainable embedding layer")

        return embedding_layer

    def create_conv_blocks(
            self,
            input_layer: tf.keras.layers.Layer
    ) -> List[tf.keras.layers.Layer]:
        """Tạo các Conv1D blocks với nhiều kernel sizes"""
        conv_blocks = []

        for filter_size in self.config['filter_sizes']:
            for dilation_rate in self.config['dilation_rates']:
                conv = Conv1D(
                    filters=self.config['num_filters'],
                    kernel_size=filter_size,
                    dilation_rate=dilation_rate,
                    padding='valid',
                    activation=None,
                    kernel_initializer='he_normal'
                )(input_layer)

                # Layer Normalization
                conv = LayerNormalization()(conv)

                # Activation
                conv = tf.keras.activations.gelu(conv)

                # Global Max Pooling
                pool = GlobalMaxPooling1D()(conv)

                conv_blocks.append(pool)

        return conv_blocks

    def build(self) -> Model:
        """Xây dựng mô hình CNN hoàn chỉnh"""
        # Input Layer
        input_layer = Input(shape=(self.config['max_sequence_length'],))

        # Embedding Layer
        embedding_layer = self.create_embedding_layer()
        x = embedding_layer(input_layer)

        # Dropout sau embedding
        x = Dropout(self.config['embedding_dropout'])(x)

        # CNN Blocks
        conv_blocks = self.create_conv_blocks(x)

        # Concatenate tất cả features
        x = Concatenate()(conv_blocks)

        # Fully Connected Layers với residual connections
        for units in self.config['dense_units']:
            # Dense block
            residual = x
            x = Dense(units, activation=None)(x)
            x = LayerNormalization()(x)
            x = tf.keras.activations.gelu(x)
            x = Dropout(self.config['dense_dropout'])(x)

            # Residual connection nếu có thể
            if residual.shape[-1] == units:
                x = tf.keras.layers.Add()([x, residual])

        # Output Layer
        output_layer = Dense(self.num_classes, activation='softmax')(x)

        # Tạo model
        model = Model(inputs=input_layer, outputs=output_layer)

        return model


class CNNTrainer:
    def __init__(
            self,
            model: Model,
            config: Dict
    ):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Tạo các callbacks cho training"""
        callbacks = []

        # Early Stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True,
            mode='min'
        )
        callbacks.append(early_stopping)

        # Learning Rate Scheduler
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config['reduce_lr_patience'],
            min_lr=1e-6,
            mode='min'
        )
        callbacks.append(reduce_lr)

        # Model Checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config['model_path'],
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        callbacks.append(checkpoint)

        # TensorBoard
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=self.config['log_dir'],
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)

        return callbacks

    def compile_model(self):
        """Compile model với optimizer và loss function"""
        # Tạo optimizer với gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            clipnorm=self.config['clip_norm']
        )

        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

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
        self.compile_model()

        # Create callbacks
        callbacks = self.create_callbacks()

        # Train model
        self.logger.info("Starting model training...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        return history.history


class CNNPredictor:
    def __init__(
            self,
            model: Model,
            config: Dict
    ):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

    def predict(
            self,
            X: np.ndarray,
            batch_size: int = None
    ) -> np.ndarray:
        """Dự đoán với model đã train"""
        if batch_size is None:
            batch_size = self.config['batch_size']

        predictions = self.model.predict(
            X,
            batch_size=batch_size,
            verbose=1
        )

        return predictions

    def evaluate(
            self,
            X: np.ndarray,
            y: np.ndarray,
            batch_size: int = None
    ) -> Dict:
        """Đánh giá model trên tập test"""
        if batch_size is None:
            batch_size = self.config['batch_size']

        results = self.model.evaluate(
            X, y,
            batch_size=batch_size,
            verbose=1
        )

        metrics = {
            'loss': results[0],
            'accuracy': results[1]
        }

        return metrics