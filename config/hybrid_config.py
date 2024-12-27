

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

HYBRID_CONFIG = {
    # Model Architecture
    'embedding_dim': 300,
    'max_sequence_length': 500,

    # CNN parameters
    'cnn_filter_sizes': [3, 4, 5],
    'num_filters': 64,

    # LSTM parameters
    'lstm_units': 64,

    # Regularization
    'embedding_dropout': 0.2,
    'dense_dropout': 0.5,

    # Training
    'batch_size': 64,
    'epochs': 10,
    'learning_rate': 0.001,
    'early_stopping_patience': 3,
    'reduce_lr_patience': 2,
    'clip_norm': 1.0,

    # Paths
    'model_path': MODELS_DIR / 'hybrid_model1.keras',
    'log_dir': 'logs/hybrid'
}