import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config.cnn_config import *
from src.models.cnn_model import TextCNN, CNNTrainer, CNNPredictor
import pickle



def setup_logging():
    """Thiết lập logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_data(processed_data_dir: Path):
    """Load dữ liệu đã xử lý"""
    train_df = pd.read_csv(processed_data_dir / 'train.csv')
    val_df = pd.read_csv(processed_data_dir / 'val.csv')
    test_df = pd.read_csv(processed_data_dir / 'test.csv')

    return train_df, val_df, test_df


def prepare_sequences(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config: dict
):
    """Chuẩn bị sequences cho model"""
    # Tokenize texts
    tokenizer = Tokenizer(num_words=config['max_features'])
    tokenizer.fit_on_texts(train_df['full_text'])

    # Convert texts to sequences
    X_train = tokenizer.texts_to_sequences(train_df['full_text'])
    X_val = tokenizer.texts_to_sequences(val_df['full_text'])
    X_test = tokenizer.texts_to_sequences(test_df['full_text'])

    # Pad sequences
    X_train = pad_sequences(X_train, maxlen=config['max_sequence_length'])
    X_val = pad_sequences(X_val, maxlen=config['max_sequence_length'])
    X_test = pad_sequences(X_test, maxlen=config['max_sequence_length'])

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['class'])
    y_val = label_encoder.transform(val_df['class'])
    y_test = label_encoder.transform(test_df['class'])



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
        # Load data
        logger.info("Loading processed data...")
        train_df, val_df, test_df = load_data(Path('/home/ponydasierra/projects/news_classification/newdata/processed'))

        # Prepare sequences
        logger.info("Preparing sequences...")
        train_data, val_data, test_data, tokenizer, label_encoder = prepare_sequences(
            train_df, val_df, test_df, CNN_CONFIG
        )

        # Calculate class weights
        class_weights = calculate_class_weights(train_data[1])

        # Create model
        logger.info("Creating CNN model...")
        cnn = TextCNN(
            config=CNN_CONFIG,
            vocab_size=len(tokenizer.word_index) + 1,
            num_classes=len(label_encoder.classes_)
        )
        model = cnn.build()

        # Create trainer
        trainer = CNNTrainer(model, CNN_CONFIG)

        # Train model
        logger.info("Training model...")
        history = trainer.train(train_data, val_data, class_weights)

        # Lấy giá trị accuracy và loss trên tập train từ history
        train_accuracy = history['accuracy'][-1]
        train_loss = history['loss'][-1]
        logger.info(f"Train Accuracy: {train_accuracy:.4f}, Train Loss: {train_loss:.4f}")

        # Evaluate on test set
        logger.info("Evaluating model...")
        predictor = CNNPredictor(model, CNN_CONFIG)
        test_metrics = predictor.evaluate(test_data[0], test_data[1])

        logger.info(f"Test metrics: {test_metrics}")

        # Save model artifacts
        logger.info("Saving model artifacts...")
        model.save(CNN_CONFIG['model_path'])

        # Lưu tokenizer và label_encoder
        logger.info("Saving tokenizer and label encoder...")

        # Lưu tokenizer
        with open(MODELS_DIR / 'tokenizer1.pickle', 'wb') as f:
            pickle.dump(tokenizer, f)

        # Lưu label encoder
        with open(MODELS_DIR / 'label_encoder1.pickle', 'wb') as f:
            pickle.dump(label_encoder, f)

        logger.info("Saved artifacts successfully!")


    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()