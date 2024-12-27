
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
CNN_CONFIG = {
    # Model Architecture
    'embedding_dim': 300,
    'max_features': 20000,  # Maximum number of words to keep, based on word frequency
    'max_sequence_length': 500,
    'filter_sizes': [2, 3, 4, 5],
    'dilation_rates': [1, 2],
    'num_filters': 128,
    'dense_units': [256, 128],

    # Regularization
    'embedding_dropout': 0.2,
    'dense_dropout': 0.5,
    'l2_reg': 0.01,
    'clip_norm': 1.0,

    # Training
    'batch_size': 64,
    'epochs': 10,
    'learning_rate': 0.001,
    'early_stopping_patience': 5,
    'reduce_lr_patience': 2,

    # Paths
    'model_path': MODELS_DIR / 'cnn_model1.keras',
    'log_dir': 'logs/cnn'
}