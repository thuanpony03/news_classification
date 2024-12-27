import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config.cnn_config import *
from config.hybrid_config import  *


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
    with open(MODELS_DIR / 'tokenizer1.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    with open(MODELS_DIR / 'label_encoder1.pickle', 'rb') as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


def prepare_test_data(test_df, tokenizer, label_encoder, config):
    """Chuẩn bị test data cho prediction"""
    # Prepare texts
    X_test = tokenizer.texts_to_sequences(test_df['full_text'])
    X_test = tf.keras.preprocessing.sequence.pad_sequences(
        X_test,
        maxlen=config['max_sequence_length']
    )

    # Prepare labels
    y_test = label_encoder.transform(test_df['class'])

    return X_test, y_test


def plot_confusion_matrix(y_true, y_pred, labels, output_dir):
    """Vẽ confusion matrix"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()


def analyze_classification_report(y_true, y_pred, labels, output_dir):
    """Tạo classification report"""
    from sklearn.metrics import classification_report
    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True
    )

    # Convert to DataFrame for better visualization
    df_report = pd.DataFrame(report).transpose()

    # Save to CSV
    df_report.to_csv(output_dir / 'classification_report.csv')

    return df_report


def analyze_errors(test_df, y_true, y_pred, label_encoder, output_dir):
    """Phân tích các trường hợp dự đoán sai"""
    # Get actual and predicted labels
    y_true_labels = label_encoder.inverse_transform(y_true)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Create error analysis DataFrame
    error_df = pd.DataFrame({
        'text': test_df['full_text'],
        'true_label': y_true_labels,
        'predicted_label': y_pred_labels,
        'is_correct': y_true_labels == y_pred_labels
    })

    # Filter incorrect predictions
    error_df = error_df[~error_df['is_correct']]

    # Save error analysis
    error_df.to_csv(output_dir / 'error_analysis.csv', index=False)

    return error_df


def main():
    # Setup
    logger = setup_logging()
    output_dir = Path('results/HYBRID_analysis1')
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load model and artifacts
        logger.info("Loading model and artifacts...")
        model, tokenizer, label_encoder = load_model_artifacts()

        # Load test data
        logger.info("Loading test data...")
        test_df = pd.read_csv('/home/ponydasierra/projects/news_classification/data/processed/test.csv')

        # Prepare test data
        logger.info("Preparing test data...")
        X_test, y_test = prepare_test_data(test_df, tokenizer, label_encoder, HYBRID_CONFIG)

        # Get predictions
        logger.info("Getting predictions...")
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Plot confusion matrix
        logger.info("Generating confusion matrix...")
        plot_confusion_matrix(
            y_test,
            y_pred,
            label_encoder.classes_,
            output_dir
        )

        # Generate classification report
        logger.info("Generating classification report...")
        report_df = analyze_classification_report(
            y_test,
            y_pred,
            label_encoder.classes_,
            output_dir
        )

        # # Analyze errors
        # logger.info("Analyzing errors...")
        # error_df = analyze_errors(
        #     test_df,
        #     y_test,
        #     y_pred,
        #     label_encoder,
        #     output_dir
        # )

        # Log results summary
        logger.info("\n=== RESULTS SUMMARY ===")
        logger.info(f"Total test samples: {len(y_test)}")
        logger.info(f"Correct predictions: {sum(y_test == y_pred)}")
        logger.info(f"Accuracy: {sum(y_test == y_pred) / len(y_test):.4f}")
        logger.info(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()