import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config.hybrid_config import HYBRID_CONFIG, PROCESSED_DATA_DIR
from src.models.hybrid_model import HybridCNNLSTM, HybridTrainer
import mlflow
import mlflow.tensorflow
from config.hybrid_config import *


def setup_logging():
    """Thiết lập logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_and_prepare_data(processed_data_dir: Path):
    """Load và chuẩn bị dữ liệu"""
    # Load data
    train_df = pd.read_csv(processed_data_dir / 'train_data.csv')
    val_df = pd.read_csv(processed_data_dir / 'val_data.csv')
    test_df = pd.read_csv(processed_data_dir / 'test_data.csv')

    # Create tokenizer
    tokenizer = Tokenizer(num_words=None)
    tokenizer.fit_on_texts(train_df['full_text'])

    # Convert texts to sequences
    X_train = tokenizer.texts_to_sequences(train_df['full_text'])
    X_val = tokenizer.texts_to_sequences(val_df['full_text'])
    X_test = tokenizer.texts_to_sequences(test_df['full_text'])

    # Pad sequences
    X_train = pad_sequences(
        X_train,
        maxlen=HYBRID_CONFIG['max_sequence_length'],
        padding='post',
        truncating='post'
    )
    X_val = pad_sequences(
        X_val,
        maxlen=HYBRID_CONFIG['max_sequence_length'],
        padding='post',
        truncating='post'
    )
    X_test = pad_sequences(
        X_test,
        maxlen=HYBRID_CONFIG['max_sequence_length'],
        padding='post',
        truncating='post'
    )

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['label'])
    y_val = label_encoder.transform(val_df['label'])
    y_test = label_encoder.transform(test_df['label'])

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), tokenizer, label_encoder


def calculate_class_weights(y_train):
    """Tính class weights cho imbalanced data"""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    return dict(zip(classes, weights))


def main():
    # Setup
    logger = setup_logging()

    try:
        # Start MLflow run
        with mlflow.start_run(run_name='hybrid_model'):
            # Log config
            mlflow.log_params(HYBRID_CONFIG)

            # Load and prepare data
            logger.info("Loading and preparing data...")
            train_data, val_data, test_data, tokenizer, label_encoder = load_and_prepare_data(
                Path('/home/ponydasierra/thuctap/news_classification/data/processed')
            )

            # Calculate class weights
            class_weights = calculate_class_weights(train_data[1])

            # Create model
            logger.info("Creating hybrid model...")
            hybrid_model = HybridCNNLSTM(
                config=HYBRID_CONFIG,
                vocab_size=len(tokenizer.word_index) + 1,
                num_classes=len(label_encoder.classes_)
            )
            model = hybrid_model.build()

            # Log model architecture
            model.summary(print_fn=logger.info)

            # Create trainer
            trainer = HybridTrainer(model, HYBRID_CONFIG)

            # Train model
            logger.info("Training model...")
            history = trainer.train(train_data, val_data, class_weights)


            # Log metrics
            for metric_name, metric_values in history.items():
                for epoch, value in enumerate(metric_values):
                    mlflow.log_metric(metric_name, value, step=epoch)


            # Lấy giá trị accuracy và loss trên tập train từ history
            train_accuracy = history['accuracy'][-1]
            train_loss = history['loss'][-1]
            logger.info(f"Train Accuracy: {train_accuracy:.4f}, Train Loss: {train_loss:.4f}")
            # Evaluate on test set
            logger.info("Evaluating model...")
            test_loss, test_acc = model.evaluate(
                test_data[0],
                test_data[1],
                verbose=1
            )

            # Log test metrics
            mlflow.log_metrics({
                "test_loss": test_loss,
                "test_accuracy": test_acc
                # Removing top-2 and top-3 accuracy metrics
            })

            # Generate predictions for analysis
            y_pred = model.predict(test_data[0])
            y_pred_classes = np.argmax(y_pred, axis=1)

            # Generate classification report
            from sklearn.metrics import classification_report
            report = classification_report(
                test_data[1],
                y_pred_classes,
                target_names=label_encoder.classes_,
                output_dict=True
            )

            # Log detailed metrics per class
            for class_name, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{class_name}_{metric_name}", value)

            # Save model artifacts
            logger.info("Saving model artifacts...")
            model.save(HYBRID_CONFIG['model_path'])

            # Save tokenizer and label encoder
            import pickle
            # Lưu tokenizer
            with open(MODELS_DIR / 'tokenizer1.pickle', 'wb') as f:
                pickle.dump(tokenizer, f)

            # Lưu label encoder
            with open(MODELS_DIR / 'label_encoder1.pickle', 'wb') as f:
                pickle.dump(label_encoder, f)

            logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
