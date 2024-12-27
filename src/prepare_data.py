import logging
from config.config import *
from src.data_preparation import DataPreparation
import pandas as pd

def setup_logging():
    """Thiết lập logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/data_preparation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main():
    # Setup logging
    logger = setup_logging()

    ROOT_DIR = Path(__file__).parent.parent
    input_file = ROOT_DIR / 'data/raw/news.csv'
    output_dir = ROOT_DIR / 'data/processed'
    output_dir.mkdir(parents=True, exist_ok=True)


    try:
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)

        prep = DataPreparation()
        train_df, val_df, test_df = prep.prepare_data(df)

        train_file = output_dir / 'train_data.csv'
        val_file = output_dir / 'val_data.csv'
        test_file = output_dir / 'test_data.csv'

        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)

        logger.info(f"Saved train data to {train_file}")
        logger.info(f"Saved validation data to {val_file}")
        logger.info(f"Saved test data to {test_file}")

    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()