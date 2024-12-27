import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import logging
from pathlib import Path

from config.cnn_config import CNN_CONFIG, ROOT_DIR
from config.config import ROOT_DIR
from config.hybrid_config import *


def setup_logging():
    """Thiết lập logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_model_artifacts():
    """Load model và các artifacts cần thiết"""
    # Load model
    model = tf.keras.models.load_model(HYBRID_CONFIG['model_path'])

    # Load tokenizer và label encoder
    with open(MODELS_DIR / 'tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    with open(MODELS_DIR / 'label_encoder.pickle', 'rb') as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


def prepare_prediction_data(texts, tokenizer, config):
    """Chuẩn bị data mới cho prediction"""
    # Convert texts to sequences
    X_pred = tokenizer.texts_to_sequences(texts)

    # Pad sequences
    X_pred = tf.keras.preprocessing.sequence.pad_sequences(
        X_pred,
        maxlen=config['max_sequence_length']
    )

    return X_pred


def predict_new_data(input_file, output_file):
    """Hàm chính để predict data mới"""
    logger = setup_logging()

    try:
        # Load model và artifacts
        logger.info("Loading model and artifacts...")
        model, tokenizer, label_encoder = load_model_artifacts()

        # Đọc data mới
        logger.info("Loading new data...")
        new_data = pd.read_csv(input_file)

        # Kiểm tra column 'full_text' có tồn tại
        if 'full_text' not in new_data.columns:
            raise ValueError("Input data must contain 'full_text' column")

        # Chuẩn bị data
        logger.info("Preparing data for prediction...")
        X_pred = prepare_prediction_data(
            new_data['full_text'],
            tokenizer,
            HYBRID_CONFIG
        )

        # Thực hiện prediction
        logger.info("Making predictions...")
        y_pred_proba = model.predict(X_pred)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Convert predictions to labels
        predicted_labels = label_encoder.inverse_transform(y_pred)

        # Thêm kết quả prediction vào DataFrame
        new_data['predicted_class'] = predicted_labels
        new_data['confidence'] = np.max(y_pred_proba, axis=1)

        # Lưu kết quả
        logger.info(f"Saving predictions to {output_file}...")
        new_data.to_csv(output_file, index=False)

        logger.info("Prediction completed successfully!")
        logger.info(f"Results saved to: {output_file}")

        # In thống kê cơ bản
        logger.info("\n=== PREDICTION SUMMARY ===")
        logger.info(f"Total samples predicted: {len(predicted_labels)}")
        logger.info("\nClass distribution:")
        for class_name, count in new_data['predicted_class'].value_counts().items():
            logger.info(f"{class_name}: {count}")

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # Định nghĩa input và output file paths
    ROOT_DIR = Path(__file__).parent.parent
    input_file = ROOT_DIR/ "new_data1/processed/processed_data.csv"
    output_file = ROOT_DIR / "new_data1/hybrid_predicted/ predictions.csv"

    predict_new_data(input_file, output_file)