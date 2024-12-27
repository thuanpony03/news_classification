import pandas as pd
import re

from sklearn.model_selection import train_test_split
from underthesea import word_tokenize
import logging
from pathlib import Path


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/prepration.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class DataPreparation:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_text(self, text: str) -> str:
        try:
            tokens = word_tokenize(text)
            return ' '.join(tokens)
        except Exception as e:
            self.logger.warning(f"Error in tokenizing text: {e}")
            return text

    def prepare_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self.logger.info("Processing titles...")
        df['cleaned_title'] = df['title'].apply(self.clean_text)
        df['tokenized_title'] = df['cleaned_title'].apply(self.tokenize_text)

        self.logger.info("Processing content...")
        df['cleaned_content'] = df['content'].apply(self.clean_text)
        df['tokenized_content'] = df['cleaned_content'].apply(self.tokenize_text)

        df['full_text'] = df['tokenized_title'] + ' ' + df['tokenized_content']
        return df

    def prepare_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['hour'] = df['date'].dt.hour

        df['domain'] = df['source'].apply(lambda x: x.split('/')[2] if isinstance(x, str) else '')
        df['user'] = df['user'].fillna('unknown')
        return df


    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        """Split data into train, validation, and test sets."""
        self.logger.info("Splitting data into train, validation, and test sets...")

        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        val_split = val_size / (1 - test_size)  # Adjust validation size relative to the train set
        train_df, val_df = train_test_split(train_df, test_size=val_split, random_state=random_state)

        self.logger.info(f"Train set size: {len(train_df)}")
        self.logger.info(f"Validation set size: {len(val_df)}")
        self.logger.info(f"Test set size: {len(test_df)}")

        return train_df, val_df, test_df

    def prepare_data(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """Prepare data and split into train, validation, and test sets."""
        df = self.prepare_text_features(df)
        df = self.prepare_metadata_features(df)
        train_df, val_df, test_df = self.split_data(df)
        return train_df, val_df, test_df
