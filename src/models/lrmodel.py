import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np
from typing import Dict, List
import logging

class LogisticRegression:
    def __init__(
            self,
            config: Dict,
            input_dim: int,
            num_classes: int
    ):
        self.config = config
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.logger = logging.getLogger(__name__)

    def build(self) -> Model:
        """Xây dựng mô hình Logistic Regression"""
        # Input Layer
        input_layer = Input(shape=(self.input_dim,))

        # Thêm Dropout để regularization (tùy chọn)
        x = Dropout(self.config.get('input_dropout', 0.2))(input_layer)

        # Output Layer với activation là softmax cho multi-class classification
        output_layer = Dense(
            self.num_classes,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(self.config.get('l2_lambda', 0.01))
        )(x)

        # Tạo model
        model = Model(inputs=input_layer, outputs=output_layer)
        return model

class LogisticRegressionTrainer:
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
            patience=self.config.get('early_stopping_patience', 5),
            restore_best_weights=True,
            mode='min'
        )
        callbacks.append(early_stopping)

        # Model Checkpoint
        if 'model_path' in self.config:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.config['model_path'],
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            )
            callbacks.append(checkpoint)

        return callbacks

    def compile_model(self):
        """Compile model với optimizer và loss function"""
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.get('learning_rate', 0.001)
        )

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
        self.logger.info("Starting Logistic Regression training...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.config.get('batch_size', 32),
            epochs=self.config.get('epochs', 50),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        return history.history

class LogisticRegressionPredictor:
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
            batch_size = self.config.get('batch_size', 32)

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
            batch_size = self.config.get('batch_size', 32)

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

# Ví dụ sử dụng:
"""
# Config cho model
config = {
    'input_dropout': 0.2,
    'l2_lambda': 0.01,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'early_stopping_patience': 5,
    'model_path': 'best_logistic_model.h5'
}

# Khởi tạo model
input_dim = X_train.shape[1]  # Số features
num_classes = len(np.unique(y_train))  # Số classes

logistic_model = LogisticRegression(config, input_dim, num_classes)
model = logistic_model.build()

# Khởi tạo trainer
trainer = LogisticRegressionTrainer(model, config)

# Train model
history = trainer.train(
    train_data=(X_train, y_train),
    val_data=(X_val, y_val)
)

# Dự đoán và đánh giá
predictor = LogisticRegressionPredictor(model, config)
predictions = predictor.predict(X_test)
metrics = predictor.evaluate(X_test, y_test)
"""