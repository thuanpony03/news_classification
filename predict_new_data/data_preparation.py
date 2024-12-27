import pandas as pd
import re
from underthesea import word_tokenize
import logging
from pathlib import Path


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/prediction_prep.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class PredictionDataPrep:
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

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare_text_features(df)
        df = self.prepare_metadata_features(df)
        return df


def main():
    logger = setup_logging()
    ROOT_DIR = Path(__file__).parent.parent
    print(ROOT_DIR)
    input_file = ROOT_DIR / 'new_data1/raw/Truong_fake_Full.csv'
    output_file = ROOT_DIR / 'new_data1/processed/processed_data.csv'
    try:
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)

        prep = PredictionDataPrep()
        processed_df = prep.prepare_data(df)

        processed_df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")

    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()