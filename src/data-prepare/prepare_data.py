import logging
from pathlib import Path
import pandas as pd
from data_preparation import DataPreparation, setup_logging

def main():
    # Setup logging
    logger = setup_logging()

    ROOT_DIR = Path(__file__).parent.parent.parent
    input_file = ROOT_DIR / 'data/raw/dataset_labeled.csv'
    output_dir = ROOT_DIR / 'data/processed5'
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)

        # Log thông tin về dữ liệu
        logger.info("\nDataset Info:")
        logger.info(f"Total samples: {len(df)}")
        logger.info("\nLabel distribution:")
        logger.info(df['label'].value_counts())

        prep = DataPreparation()
        train_df, val_df, test_df, df_cleaned = prep.prepare_data(df)

        train_file = output_dir / 'train_data.csv'
        val_file = output_dir / 'val_data.csv'
        test_file = output_dir / 'test_data.csv'
        clean_file = output_dir / 'clean_data.csv'

        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
        df_cleaned.to_csv(clean_file, index=False)

        logger.info(f"Saved train data to {train_file}")
        logger.info(f"Saved validation data to {val_file}")
        logger.info(f"Saved test data to {test_file}")
        logger.info(f"Saved clean data to {df_cleaned}")


    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()