from pathlib import Path

# Đường dẫn
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"


# Data preparation config
DATA_CONFIG = {
    'train_size': 0.8,
    'val_size': 0.1,
    'test_size': 0.1,
    'random_state': 42,
    'min_text_length': 10,  # Lọc văn bản quá ngắn
    'max_text_length': 1000,  # Lọc văn bản quá dài
}

# File paths
DATA_PATHS = {
    'raw_data': 'data/raw/news_data.csv',
    'processed_data': 'data/processed',
    'models': 'models',
    'logs': 'logs'
}